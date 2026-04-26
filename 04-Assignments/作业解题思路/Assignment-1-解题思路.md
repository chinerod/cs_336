# Assignment 1 解题思路指南

## 📝 作业概述

**主题**: Transformer架构与语言模型训练
**目标**: 从零实现完整的Transformer语言模型，并训练出能生成文本的模型
**难度**: ⭐⭐⭐（基础但代码量大）
**预计时间**: 2-3周

---

## 📚 知识准备

### 理论准备
1. **BPE Tokenization** - 理解subword算法
2. **Transformer架构** - Attention机制、LayerNorm、残差连接
3. **语言建模** - 自回归模型、交叉熵损失
4. **PyTorch** - nn.Module、DataLoader、训练循环

### 推荐阅读
- Attention Is All You Need（必读）
- CS336 Week 1-4 Lecture Notes
- The Illustrated Transformer（博客）

---

## 🗂️ 作业结构

### Part 1: Transformer架构实现（50%）

#### 1.1 Tokenizer
**要求**: 实现BPE Tokenizer

**核心函数**:
```python
class BPETokenizer:
    def train(self, text: str, vocab_size: int) -> None:
        """训练BPE词表"""
        # 1. 初始化词表（所有字符）
        # 2. 统计词频
        # 3. 迭代合并最频繁的pair
        # 4. 保存merge规则

    def encode(self, text: str) -> list[int]:
        """文本→token IDs"""
        # 1. 按空格分词
        # 2. 对每个词应用merge规则
        # 3. 转换为token IDs

    def decode(self, token_ids: list[int]) -> str:
        """token IDs→文本"""
        # 反向操作
```

**实现提示**:
- 使用贪心算法找最频繁pair
- 注意 `<|endoftext|>` 特殊token
- 处理未知字符

**测试方法**:
```python
tokenizer = BPETokenizer()
tokenizer.train(corpus, vocab_size=5000)
ids = tokenizer.encode("Hello world")
text = tokenizer.decode(ids)
assert tokenizer.encode(text) == ids
```

#### 1.2 Multi-Head Attention
**要求**: 实现Scaled Dot-Product Attention和Multi-Head Attention

**核心公式**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

**实现步骤**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        # d_model必须能被n_heads整除
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

    def forward(self, x, mask=None):
        # 1. 线性投影得到Q, K, V
        # 2. reshape成多头的形式
        # 3. 计算attention scores
        # 4. 应用causal mask（对于语言模型）
        # 5. softmax
        # 6. 加权求和
        # 7. 输出投影
```

**关键细节**:
- 缩放因子 `1/√d_k` 很重要
- Causal mask防止看到未来信息
- 可以用 `torch.triu` 生成mask

**代码框架**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal_mask=True):
        B, T, C = x.shape

        # 投影并reshape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if causal_mask:
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
            scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # reshape并投影
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```

#### 1.3 Transformer Block
**要求**: 实现完整的Transformer Block

**结构**:
```
Input → LayerNorm → MultiHeadAttention → Residual →
        LayerNorm → FeedForward → Residual → Output
```

**实现**:
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention + Residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # FeedForward + Residual
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x
```

#### 1.4 完整Transformer LM
**要求**: 组装完整模型

**架构**:
```
Token Embedding → Position Embedding → Transformer Blocks × N →
LayerNorm → Linear → Softmax
```

**实现**:
```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        B, T = token_ids.shape

        # Embedding
        tok_emb = self.token_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=token_ids.device))
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
```

#### 1.5 Generate函数
**要求**: 实现自回归生成

**实现**:
```python
def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
    """自回归生成文本"""
    self.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 截断到最大长度
            if prompt.size(1) >= self.max_len:
                break

            # 前向传播
            logits = self(prompt)
            logits = logits[:, -1, :]  # 取最后一个token的logits

            # 温度缩放
            logits = logits / temperature

            # Top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接到prompt
            prompt = torch.cat([prompt, next_token], dim=1)

    return prompt
```

---

### Part 2: 训练基础设施（50%）

#### 2.1 数据加载
**要求**: 实现Dataset和DataLoader

**Dataset实现**:
```python
class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 加载并tokenize数据
        with open(data_path, 'r') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) // self.max_len

    def __getitem__(self, idx):
        start = idx * self.max_len
        end = start + self.max_len + 1
        chunk = self.tokens[start:end]

        # input和target错位一位
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y
```

#### 2.2 训练循环
**要求**: 实现完整的训练流程

**训练函数**:
```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)
```

#### 2.3 学习率调度
**要求**: 实现Warmup + Cosine Decay

```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
```

#### 2.4 验证与日志
**要求**: 实现验证循环和日志记录

```python
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

# 计算perplexity
def compute_perplexity(loss):
    return math.exp(loss)
```

---

## 🎯 完整训练流程

### 超参数配置（124M模型）
```python
config = {
    'vocab_size': 5000,
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 12,
    'd_ff': 3072,
    'max_len': 1024,
    'dropout': 0.1,
    'batch_size': 16,
    'learning_rate': 3e-4,
    'warmup_steps': 2000,
    'max_steps': 100000,
}
```

### 训练脚本框架
```python
def main():
    # 1. 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=5000)

    # 2. 准备数据
    train_dataset = TextDataset('train.txt', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. 创建模型
    model = TransformerLM(**config).to(device)

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, 2000, 100000, 3e-4)

    # 5. 训练循环
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        print(f"Perplexity: {compute_perplexity(val_loss):.2f}")

        # 保存checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')

        # 生成样本文本
        sample = generate_sample(model, tokenizer, "The future of AI")
        print(f"Sample: {sample}")
```

---

## 🔍 调试技巧

### 1. 从小模型开始测试
```python
# 先测试tiny模型
tiny_config = {
    'vocab_size': 100,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 256,
}
```

### 2. 检查梯度流
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")
```

### 3. 检查输出范围
```python
# 确保loss合理（ln(vocab_size) ≈ 8.5是随机猜测水平）
print(f"Random guess loss: {math.log(vocab_size):.2f}")
```

### 4. 可视化attention
```python
# 返回attention weights，可视化看是否合理
```

---

## ⚠️ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| Loss不下降 | 学习率太大/太小，代码bug | 检查学习率，检查forward/backward |
| NaN loss | 梯度爆炸 | 使用梯度裁剪，检查初始化 |
| 内存不足 | batch size太大 | 减小batch，使用梯度累积 |
| 生成文本混乱 | 温度太低/太高 | 调整temperature |
| 输出维度错误 | reshape问题 | 检查view/permute后的shape |

---

## 📊 评估标准

- [ ] BPE Tokenizer正确编码解码
- [ ] Multi-Head Attention数值正确
- [ ] 模型输出shape正确
- [ ] Loss随训练下降
- [ ] Perplexity < 50（在WikiText上）
- [ ] 能生成连贯文本

---

## 💡 进阶建议

1. **使用混合精度训练** - 加速训练
2. **添加gradient checkpointing** - 节省显存
3. **实现beam search** - 更好的生成
4. **添加WandB日志** - 监控训练

---

**祝你Assignment 1顺利完成！** 💪

