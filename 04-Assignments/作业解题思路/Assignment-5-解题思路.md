# Assignment 5 解题思路指南

## 📝 作业概述

**主题**: RLHF（Reinforcement Learning from Human Feedback）
**目标**: 实现SFT、Reward Model训练和PPO/DPO
**难度**: ⭐⭐⭐⭐（理论+实现）
**预计时间**: 2-3周
**核心技能**: RL算法、偏好学习、模型对齐

---

## 📚 知识准备

### RLHF三阶段

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   预训练     │ ──> │    SFT     │ ──> │    RL      │
│  (已完成)    │     │   微调     │     │   对齐     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                     │
                           v                     v
                    ┌─────────────┐     ┌─────────────┐
                    │  指令数据    │     │ Reward Model│
                    └─────────────┘     └─────────────┘
```

### 核心概念

#### 1. SFT（Supervised Fine-Tuning）
- 用高质量的指令-回复对微调模型
- 让模型学会遵循指令

#### 2. Reward Model
- 学习人类的偏好
- 输入：prompt + response
- 输出：scalar reward

#### 3. RL（PPO/DPO）
- 用Reward Model指导模型生成
- 优化目标：高reward + 不偏离SFT模型太远

---

## 🗂️ 作业结构

### Part 1: SFT实现

#### 1.1 准备数据

**数据格式**:
```json
{
    "prompt": "解释什么是机器学习",
    "response": "机器学习是一种人工智能的分支..."
}
```

**数据处理**:
```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path) as f:
            self.data = [json.loads(line) for line in f]

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建完整文本
        # 格式: <prompt>\n<response>
        full_text = f"{item['prompt']}\n{item['response']}"

        # Tokenize
        tokens = self.tokenizer.encode(full_text)

        # 截断
        tokens = tokens[:self.max_length]

        # input和target
        input_ids = tokens[:-1]
        labels = tokens[1:]

        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels)
        }

    def __len__(self):
        return len(self.data)
```

#### 1.2 SFT训练

```python
class SFTTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=0.01
        )

    def train(self, train_loader, num_epochs):
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                # 前向传播
                logits = self.model(input_ids)

                # 计算loss（只在response部分）
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction='mean'
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
```

---

### Part 2: Reward Model实现

#### 2.1 数据准备

**偏好数据格式**:
```json
{
    "prompt": "解释什么是机器学习",
    "chosen": "机器学习是一种人工智能的分支...（更好的回答）",
    "rejected": "机器学习就是机器自己学习...（较差的回答）"
}
```

#### 2.2 Reward Model架构

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # 替换head为reward head
        d_model = base_model.d_model
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        # 通过base model
        hidden = self.base_model(input_ids, return_hidden=True)

        # 取最后一个token的hidden state
        last_hidden = hidden[:, -1, :]

        # 计算reward
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)
```

#### 2.3 Bradley-Terry损失

```python
class RewardModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr']
        )

    def compute_loss(self, batch):
        """
        Bradley-Terry损失
        最大化 P(chosen > rejected)
        """
        prompts = batch['prompt']
        chosen = batch['chosen']
        rejected = batch['rejected']

        # Tokenize
        chosen_ids = self.tokenizer.encode(prompts + chosen)
        rejected_ids = self.tokenizer.encode(prompts + rejected)

        # 计算reward
        chosen_reward = self.model(chosen_ids)
        rejected_reward = self.model(rejected_ids)

        # Bradley-Terry损失: -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

        return loss

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0

            for batch in train_loader:
                loss = self.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
```

---

### Part 3: PPO实现

#### 3.1 PPO核心概念

```
目标: 最大化reward，同时不偏离SFT模型太远

Loss = -min(
    r_t * advantage,
    clip(r_t, 1-eps, 1+eps) * advantage
) - beta * KL_penalty

其中:
- r_t = pi_new / pi_old (概率比)
- advantage = reward - value (优势函数)
- KL_penalty = KL(pi_new || pi_sft)
```

#### 3.2 PPO Trainer

```python
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, config):
        self.policy = policy_model          # 当前策略
        self.ref_model = ref_model          # SFT参考模型（冻结）
        self.reward_model = reward_model    # Reward模型（冻结）
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config['lr']
        )

    def generate_responses(self, prompts, max_length=256):
        """生成回复"""
        self.policy.eval()

        responses = []
        log_probs = []

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)

            # 自回归生成
            generated = input_ids.copy()
            seq_log_probs = []

            for _ in range(max_length):
                with torch.no_grad():
                    logits = self.policy(torch.tensor([generated]))
                    probs = F.softmax(logits[:, -1], dim=-1)

                # 采样
                next_token = torch.multinomial(probs[0], 1).item()
                generated.append(next_token)

                # 记录log_prob
                seq_log_probs.append(torch.log(probs[0, next_token]))

                if next_token == self.tokenizer.eos_token_id:
                    break

            response = self.tokenizer.decode(generated[len(input_ids):])
            responses.append(response)
            log_probs.append(torch.stack(seq_log_probs))

        return responses, log_probs

    def compute_rewards(self, prompts, responses):
        """计算reward"""
        rewards = []

        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            input_ids = self.tokenizer.encode(full_text)

            with torch.no_grad():
                reward = self.reward_model(torch.tensor([input_ids]))

            # 加上KL惩罚
            # KL(pi || pi_ref) = log(pi/pi_ref)
            with torch.no_grad():
                policy_logits = self.policy(torch.tensor([input_ids]))
                ref_logits = self.ref_model(torch.tensor([input_ids]))

                policy_logprob = F.log_softmax(policy_logits, dim=-1)
                ref_logprob = F.log_softmax(ref_logits, dim=-1)

                kl_div = (policy_logprob - ref_logprob).sum()

            # 最终reward = reward_model_score - beta * KL
            final_reward = reward - self.config['kl_beta'] * kl_div
            rewards.append(final_reward)

        return rewards

    def train_step(self, prompts, old_responses, old_log_probs, rewards):
        """PPO更新"""
        self.policy.train()

        # 重新计算log probs和advantages
        advantages = self._compute_advantages(rewards)

        for _ in range(self.config['ppo_epochs']):
            # 新策略的log prob
            new_log_probs = self._compute_log_probs(prompts, old_responses)

            # 概率比
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config['clip_eps'],
                               1 + self.config['clip_eps']) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # KL惩罚（可选，也可直接用reward里的）
            kl_loss = self._compute_kl_loss(prompts, old_responses)

            loss = policy_loss + self.config['kl_coef'] * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, train_prompts, num_iterations):
        """完整训练流程"""
        for iteration in range(num_iterations):
            # 1. 生成回复
            responses, log_probs = self.generate_responses(train_prompts)

            # 2. 计算reward
            rewards = self.compute_rewards(train_prompts, responses)

            # 3. PPO更新
            self.train_step(train_prompts, responses, log_probs, rewards)

            print(f"Iteration {iteration}: avg_reward = {sum(rewards)/len(rewards):.4f}")
```

---

### Part 4: DPO实现（可选但推荐）

#### 4.1 DPO原理

DPO（Direct Preference Optimization）不需要显式训练Reward Model：

```
DPO损失 = -log(sigmoid(beta * (log(pi/pi_ref)_chosen - log(pi/pi_ref)_rejected)))

优点:
- 不需要训练Reward Model
- 更简单高效
- 效果通常比PPO好
```

#### 4.2 DPO Trainer

```python
class DPOTrainer:
    def __init__(self, policy_model, ref_model, config):
        self.policy = policy_model
        self.ref_model = ref_model  # SFT模型，冻结
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config['lr']
        )

    def compute_loss(self, batch):
        """
        DPO损失
        """
        prompts = batch['prompt']
        chosen = batch['chosen']
        rejected = batch['rejected']

        # 构造完整序列
        chosen_full = [p + c for p, c in zip(prompts, chosen)]
        rejected_full = [p + r for p, r in zip(prompts, rejected)]

        # Tokenize
        chosen_ids = [self.tokenizer.encode(t) for t in chosen_full]
        rejected_ids = [self.tokenizer.encode(t) for t in rejected_full]

        # 计算策略模型的log prob
        policy_chosen_logps = self._get_batch_logps(self.policy, chosen_ids)
        policy_rejected_logps = self._get_batch_logps(self.policy, rejected_ids)

        # 计算参考模型的log prob
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(self.ref_model, chosen_ids)
            ref_rejected_logps = self._get_batch_logps(self.ref_model, rejected_ids)

        # DPO损失
        beta = self.config['beta']

        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps

        logits = beta * (policy_ratio - ref_ratio)
        loss = -F.logsigmoid(logits).mean()

        return loss

    def _get_batch_logps(self, model, input_ids_list):
        """计算batch的log probs"""
        logps = []

        for input_ids in input_ids_list:
            logits = model(torch.tensor([input_ids]))
            log_probs = F.log_softmax(logits, dim=-1)

            # 取实际token的log prob
            target_logps = log_probs[0, :-1].gather(1, torch.tensor(input_ids[1:]).unsqueeze(1))
            seq_logp = target_logps.sum()

            logps.append(seq_logp)

        return torch.stack(logps)

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0

            for batch in train_loader:
                loss = self.compute_loss(batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
```

---

## 📊 评估方法

### 1. Reward评估

```python
def evaluate_reward(model, reward_model, test_data):
    """评估生成质量"""
    rewards = []

    for item in test_data:
        prompt = item['prompt']

        # 生成回复
        response = generate(model, prompt)

        # 计算reward
        full = prompt + response
        reward = reward_model(tokenizer.encode(full))
        rewards.append(reward)

    return sum(rewards) / len(rewards)
```

### 2. 对比评估

```python
def compare_models(sft_model, rlhf_model, test_prompts, reward_model):
    """对比SFT和RLHF模型"""
    results = []

    for prompt in test_prompts:
        sft_response = generate(sft_model, prompt)
        rlhf_response = generate(rlhf_model, prompt)

        sft_reward = reward_model(prompt + sft_response)
        rlhf_reward = reward_model(prompt + rlhf_response)

        results.append({
            'prompt': prompt,
            'sft': sft_response,
            'rlhf': rlhf_response,
            'sft_reward': sft_reward,
            'rlhf_reward': rlhf_reward,
            'improvement': rlhf_reward > sft_reward
        })

    return results
```

### 3. 人工评估

```python
def human_evaluation(responses, criteria):
    """
    criteria: ['helpfulness', 'harmlessness', 'honesty']
    打分1-5
    """
    pass
```

---

## ⚠️ 常见陷阱

| 问题 | 原因 | 解决 |
|------|------|------|
| 模型崩溃 | KL惩罚太小 | 增大kl_beta |
| Reward hacking | Reward Model过拟合 | 多样化训练数据 |
| 训练不稳定 | 学习率太大 | 降低lr，增加clip |
| 内存不足 | KV cache占用大 | 减小batch，gradient checkpointing |
| 生成质量下降 | 过度优化 | Early stopping |

---

## ✅ 提交检查清单

- [ ] SFT模型训练完成
- [ ] Reward Model训练完成
- [ ] PPO或DPO训练完成
- [ ] RLHF模型相比SFT有提升
- [ ] Reward分数提高
- [ ] 代码有清晰注释
- [ ] 包含评估代码

---

## 💡 进阶方向

1. **Iterative RLHF** - 多轮迭代
2. **Constitutional AI** - 用规则指导
3. **RLAIF** - AI反馈替代人类
4. **Multi-objective** - 多目标优化

---

**RLHF是大模型对齐的核心技术，掌握它！** 🎯

