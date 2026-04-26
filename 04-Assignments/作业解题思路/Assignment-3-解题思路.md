# Assignment 3 解题思路指南

## 📝 作业概述

**主题**: Scaling Laws（规模定律）
**目标**: 通过实验验证模型规模、数据量与性能的关系
**难度**: ⭐⭐⭐（理论理解+实验设计）
**预计时间**: 1-2周
**核心技能**: 实验设计、数据分析、幂律拟合

---

## 📚 知识准备

### 核心理论

#### 1. Scaling Laws基本公式

**损失与计算量的关系**:
```
L(N) = (N / N_c)^(-α_N)      # 模型规模（参数N）
L(D) = (D / D_c)^(-α_D)      # 数据量（token数D）
L(C) = (C / C_c)^(-α_C)      # 计算量（FLOPs C）
```

其中：
- L: 交叉熵损失
- N: 模型参数量
- D: 训练token数
- C: 计算量（FLOPs）
- α: 幂律指数（通常α_N ≈ 0.076, α_D ≈ 0.095）
- N_c, D_c, C_c: 常数

#### 2. Chinchilla Optimal

**核心发现**:
- 模型参数N和数据量D应该同时增长
- 最优比例: D ≈ 20N
- 即每参数需要20个token

**计算公式**:
```
给定计算预算C，最优配置:
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

#### 3. 计算量估算

**训练FLOPs估算**:
```
C ≈ 6 * N * D

其中:
- N: 参数量
- D: token数
- 6: 常数（2 for forward, 4 for backward）
```

---

## 🗂️ 作业结构

### Part 1: 实验设计

#### 1.1 确定实验变量

**独立变量**:
```python
model_sizes = [
    {'n_layers': 2, 'd_model': 128, 'n_heads': 2},    # ~1M params
    {'n_layers': 4, 'd_model': 256, 'n_heads': 4},    # ~10M params
    {'n_layers': 6, 'd_model': 384, 'n_heads': 6},    # ~30M params
    {'n_layers': 8, 'd_model': 512, 'n_heads': 8},    # ~80M params
    {'n_layers': 12, 'd_model': 768, 'n_heads': 12},  # ~300M params
]

data_sizes = [
    10_000_000,     # 10M tokens
    50_000_000,     # 50M tokens
    100_000_000,    # 100M tokens
    500_000_000,    # 500M tokens
]
```

#### 1.2 控制变量

**必须控制的变量**:
- 学习率调度（按token数比例）
- Batch size（或按梯度累积控制）
- Optimizer（AdamW）
- 数据分布（相同来源）
- 训练步数（或token数）

**学习率调度策略**:
```python
def get_lr_schedule(optimizer, warmup_steps, max_steps, max_lr):
    """
    不同规模的模型使用相同的warmup比例和decay策略
    """
    warmup_steps = int(0.01 * max_steps)  # 1% warmup
    return CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps)
```

---

### Part 2: 参数计算

#### 2.1 计算模型参数量

```python
def count_parameters(n_layers, d_model, n_heads, vocab_size=5000):
    """
    计算Transformer参数量
    """
    # Embedding
    token_emb = vocab_size * d_model
    pos_emb = 1024 * d_model  # max_len

    # Transformer blocks
    # Attention: 4 projections (Q, K, V, O)
    attn_params = 4 * (d_model * d_model)

    # FFN: d_model -> 4*d_model -> d_model (standard)
    ffn_params = d_model * (4 * d_model) + (4 * d_model) * d_model

    # LayerNorm: 2 per block (pre-attn, pre-ffn)
    ln_params = 2 * (2 * d_model)  # scale + bias

    block_params = attn_params + ffn_params + ln_params

    # Final LayerNorm + LM Head
    final_params = 2 * d_model + d_model * vocab_size

    total = token_emb + pos_emb + n_layers * block_params + final_params

    return {
        'total': total,
        'embedding': token_emb + pos_emb,
        'transformer': n_layers * block_params,
        'head': final_params,
    }
```

#### 2.2 验证计算
```python
# 测试：GPT-2 small (124M)
params = count_parameters(n_layers=12, d_model=768, n_heads=12, vocab_size=50257)
print(f"参数: {params['total']:,} ({params['total']/1e6:.1f}M)")
# 应该接近124M
```

---

### Part 3: 实验运行

#### 3.1 实验脚本框架

```python
class ScalingExperiment:
    def __init__(self, model_config, data_config, train_config):
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config

    def run(self):
        """运行单个实验"""
        # 1. 创建模型
        model = TransformerLM(**self.model_config)
        n_params = sum(p.numel() for p in model.parameters())

        # 2. 加载数据
        train_loader = self._create_dataloader(self.data_config)

        # 3. 训练
        metrics = self._train(model, train_loader)

        # 4. 评估
        final_loss = self._evaluate(model)

        return {
            'n_params': n_params,
            'n_tokens': self.data_config['n_tokens'],
            'final_loss': final_loss,
            'metrics': metrics,
        }

    def _train(self, model, train_loader):
        """训练循环"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config['lr'],
            weight_decay=0.01
        )

        losses = []
        tokens_seen = 0

        for epoch in range(self.train_config['epochs']):
            for batch in train_loader:
                # 训练...
                loss = ...
                losses.append(loss.item())

                tokens_seen += batch.size(0) * batch.size(1)

                if tokens_seen >= self.data_config['n_tokens']:
                    break

        return {'losses': losses, 'tokens_seen': tokens_seen}
```

#### 3.2 批量运行实验

```python
def run_scaling_experiments():
    """批量运行所有实验"""
    results = []

    for model_size in model_sizes:
        for data_size in data_sizes:
            print(f"Running: {model_size} with {data_size} tokens")

            exp = ScalingExperiment(
                model_config=model_size,
                data_config={'n_tokens': data_size},
                train_config={'lr': 3e-4, 'epochs': 10}
            )

            result = exp.run()
            results.append(result)

            # 保存中间结果
            save_checkpoint(result)

    return results
```

---

### Part 4: 数据分析与拟合

#### 4.1 数据整理

```python
import pandas as pd
import numpy as np

def process_results(results):
    """处理实验结果"""
    df = pd.DataFrame(results)

    # 计算计算量
    df['flops'] = 6 * df['n_params'] * df['n_tokens']

    # 转换为便于拟合的格式
    df['log_n'] = np.log10(df['n_params'])
    df['log_d'] = np.log10(df['n_tokens'])
    df['log_c'] = np.log10(df['flops'])
    df['log_loss'] = np.log10(df['final_loss'])

    return df
```

#### 4.2 幂律拟合

```python
from scipy.optimize import curve_fit

def power_law(x, a, alpha):
    """幂律函数: y = a * x^(-alpha)"""
    return a * (x ** (-alpha))

def fit_scaling_law(df, x_col, y_col):
    """
    拟合Scaling Law

    拟合: loss = a * x^(-alpha)
    对数形式: log(loss) = log(a) - alpha * log(x)
    """
    x = df[x_col].values
    y = df[y_col].values

    # 拟合
    popt, pcov = curve_fit(power_law, x, y, p0=[1.0, 0.1])

    a, alpha = popt
    return {'a': a, 'alpha': alpha, 'cov': pcov}

# 拟合
n_fit = fit_scaling_law(df[df['n_tokens'] == fixed_tokens], 'n_params', 'final_loss')
d_fit = fit_scaling_law(df[df['n_params'] == fixed_params], 'n_tokens', 'final_loss')
c_fit = fit_scaling_law(df, 'flops', 'final_loss')

print(f"N scaling: L = {n_fit['a']:.4f} * N^(-{n_fit['alpha']:.4f})")
print(f"D scaling: L = {d_fit['a']:.4f} * D^(-{d_fit['alpha']:.4f})")
print(f"C scaling: L = {c_fit['a']:.4f} * C^(-{c_fit['alpha']:.4f})")
```

#### 4.3 可视化

```python
import matplotlib.pyplot as plt

def plot_scaling_curves(df, fits):
    """绘制Scaling曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # N scaling
    ax = axes[0]
    ax.scatter(df['n_params'], df['final_loss'])
    x_range = np.logspace(np.log10(df['n_params'].min()), np.log10(df['n_params'].max()), 100)
    ax.plot(x_range, power_law(x_range, **fits['n']), 'r-', label=f"α={fits['n']['alpha']:.3f}")
    ax.set_xlabel('Parameters (N)')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Scaling with Model Size')

    # D scaling
    ax = axes[1]
    ax.scatter(df['n_tokens'], df['final_loss'])
    x_range = np.logspace(np.log10(df['n_tokens'].min()), np.log10(df['n_tokens'].max()), 100)
    ax.plot(x_range, power_law(x_range, **fits['d']), 'r-', label=f"α={fits['d']['alpha']:.3f}")
    ax.set_xlabel('Tokens (D)')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Scaling with Data Size')

    # C scaling
    ax = axes[2]
    ax.scatter(df['flops'], df['final_loss'])
    x_range = np.logspace(np.log10(df['flops'].min()), np.log10(df['flops'].max()), 100)
    ax.plot(x_range, power_law(x_range, **fits['c']), 'r-', label=f"α={fits['c']['alpha']:.3f}")
    ax.set_xlabel('Compute (FLOPs)')
    ax.set_ylabel('Loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Scaling with Compute')

    plt.tight_layout()
    plt.savefig('scaling_curves.png')
```

---

### Part 5: Chinchilla分析

#### 5.1 最优配置计算

```python
def compute_chinchilla_optimal(compute_budget, alpha=0.5):
    """
    给定计算预算，计算最优模型规模和数据量

    C = 6 * N * D
    最优: N ∝ C^0.5, D ∝ C^0.5
    """
    # 假设 C_opt = 6 * N_opt * D_opt
    # 且 D_opt = 20 * N_opt (Chinchilla)
    # 则: C_opt = 6 * N_opt * 20 * N_opt = 120 * N_opt^2
    # 所以: N_opt = sqrt(C_opt / 120)

    N_opt = (compute_budget / 120) ** 0.5
    D_opt = 20 * N_opt

    return {
        'n_params': int(N_opt),
        'n_tokens': int(D_opt),
    }

# 示例
budget = 1e18  # 1 exaFLOP (GPT-3级别)
optimal = compute_chinchilla_optimal(budget)
print(f"最优配置: {optimal['n_params']/1e6:.1f}M 参数, {optimal['n_tokens']/1e9:.1f}B tokens")
```

#### 5.2 对比实验

```python
def compare_chinchilla(configs):
    """
    对比不同配置的效率

    configs: [
        {'name': 'small_data', 'n_params': 1e9, 'n_tokens': 10e9},
        {'name': 'chinchilla', 'n_params': 1e9, 'n_tokens': 20e9},
        {'name': 'large_data', 'n_params': 1e9, 'n_tokens': 100e9},
    ]
    """
    results = []

    for config in configs:
        # 计算计算量
        flops = 6 * config['n_params'] * config['n_tokens']

        # 训练并评估
        loss = train_and_evaluate(config)

        results.append({
            'name': config['name'],
            'n_params': config['n_params'],
            'n_tokens': config['n_tokens'],
            'flops': flops,
            'loss': loss,
            'efficiency': 1 / (loss * flops),  # 越高越好
        })

    return pd.DataFrame(results)
```

---

### Part 6: 预测与验证

#### 6.1 外推预测

```python
def predict_loss(n_params, fit_result):
    """
    用拟合的scaling law预测新模型的loss
    """
    a, alpha = fit_result['a'], fit_result['alpha']
    loss = a * (n_params ** (-alpha))
    return loss

# 预测100B模型的性能
predicted_loss = predict_loss(100e9, n_fit)
print(f"预测100B模型loss: {predicted_loss:.4f}")
```

#### 6.2 验证预测准确性

```python
def validate_prediction(train_sizes, test_sizes):
    """
    用小模型数据拟合，预测大模型性能
    """
    # 用小模型拟合
    small_data = [r for r in results if r['n_params'] <= 100e6]
    fit = fit_scaling_law(small_data, 'n_params', 'final_loss')

    # 预测大模型
    predictions = []
    for size in test_sizes:
        pred = predict_loss(size, fit)

        # 实际训练大模型
        actual = train_and_evaluate({'n_params': size})

        predictions.append({
            'n_params': size,
            'predicted': pred,
            'actual': actual,
            'error': abs(pred - actual) / actual,
        })

    return predictions
```

---

## ⚠️ 常见陷阱

### 1. 数据污染
**问题**: 训练集和测试集重叠
**解决**: 确保数据划分严格

### 2. 训练不充分
**问题**: 小数据量时，模型可能没有收敛
**解决**: 检查loss曲线是否平稳

### 3. 学习率不匹配
**问题**: 不同规模使用相同学习率
**解决**: 按模型规模调整或使用lr scaling laws

### 4. 拟合过度
**问题**: 用太多参数拟合太少数据点
**解决**: 确保数据点数量 > 参数数量

---

## ✅ 提交检查清单

- [ ] 至少5个不同规模的模型
- [ ] 至少4个不同数据量
- [ ] 参数计算正确
- [ ] 成功拟合幂律曲线
- [ ] 可视化图表清晰
- [ ] 计算Chinchilla最优配置
- [ ] 包含预测与验证

---

## 💡 进阶方向

1. **Emergent Abilities** - 研究能力突现
2. **Scaling of RLHF** - RLHF的scaling
3. **Scaling of Multimodal** - 多模态模型的scaling
4. **Inference Scaling** - 推理时的scaling

---

**Scaling Laws是理解大模型的核心，认真做实验！** 📊

