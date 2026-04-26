# 综述论文精选

## 📚 大模型领域综述

### 1. 预训练语言模型综述

**"Pre-trained Models for Natural Language Processing: A Survey"** (Qiu et al., 2020)

**核心内容**：
- 预训练任务分类
- 预训练模型演进
- 微调策略
- 应用任务

**关键图表**：
- PTMs时间线
- 模型对比表

---

### 2. Transformer架构综述

**"Efficient Transformers: A Survey"** (Tay et al., 2020)

**核心内容**：
- 高效注意力变体
- 线性注意力
- 稀疏注意力
- 记忆压缩

**分类**：
1. Fixed patterns (local, strided)
2. Combination of patterns (Longformer)
3. Learnable patterns (Reformer)
4. Low-rank methods (Linformer)
5. Kernel methods (Performer)

---

### 3. Scaling Laws综述

**"Scaling Laws for Language Models"** (Kaplan et al., 2020)

**核心发现**：
- Loss与参数量的幂律关系
- Loss与计算量的幂律关系
- 最优batch size
- 最优学习率

**公式**：
$$L(N) = \left(\frac{N}{N_c}\right)^{-\alpha_N}$$

---

### 4. 指令微调综述

**"Scaling Instruction-Finetuned Language Models"** (Chung et al., 2022)

**发现**：
- 任务数量 > 任务多样性 > 实例数量
- 混合比例重要
- 链式思维数据提升推理

---

### 5. 对齐技术综述

**"RLHF: A Survey"**

**内容**：
- RLHF流程
- Reward建模
- PPO算法细节
- 替代方法（DPO, RLAI）

---

### 6. 高效推理综述

**"A Survey on Large Language Model Inference"**

**主题**：
- 量化
- 剪枝
- 投机解码
- KV Cache优化
- 服务系统

---

## 📊 各领域专题

### 多模态大模型

**"Multimodal Large Language Models: A Survey"**
- 视觉-语言模型
- 音频-语言模型
- 统一架构

### 长文本处理

**"Advancing Transformer Architecture in Long Context"**
- 位置编码改进
- 上下文压缩
- 记忆机制

### 代码大模型

**"Large Language Models for Code: A Survey"**
- 代码预训练
- 代码生成
- 程序修复

---

## 🎯 阅读建议

### 按阶段阅读

**入门阶段**：
1. Transformer原始论文
2. BERT/GPT论文
3. Scaling Laws

**进阶阶段**：
1. 高效Transformer综述
2. 对齐技术综述
3. 推理优化综述

**研究阶段**：
1. 特定领域最新综述
2. 顶级会议论文集
3. ArXiv最新工作

---

## 📖 延伸阅读

### 经典综述

- Attention Is All You Need
- Deep Learning (LeCun et al., 2015)
- Natural Language Processing (Young et al., 2018)

### 前沿综述

- Mixture of Experts
- Retrieval-Augmented Generation
- Agent Systems

---

*综述论文清单 - 持续更新*

