# CS336 课程概览 - PhD研究型指南

## 🎓 课程定位

Stanford CS336 是一门**研究导向**的研究生课程，面向希望深入理解大语言模型系统的学生和研究者。

### 与其他课程的区别

| 课程 | 侧重点 | 深度 |
|------|--------|------|
| **CS336** | LLM系统（训练、优化、数据） | 工程+系统 |
| **CS224N** | NLP基础 | 算法 |
| **CS229** | 机器学习 | 理论 |
| **CS25** | Transformer专题 | 前沿研究 |

---

## 📖 课程大纲（研究版）

### Week 0-1: Foundations & Tokenization
**研究重点**: BPE的理论局限性与新方向
- BPE与字粒度方法的比较
- 多语言tokenization挑战
- Unigram、WordPiece、SentencePiece对比
- **研究方向**: 自适应vocabulary、视觉tokenization

### Week 2-3: PyTorch Systems & Transformers
**研究重点**: 深度学习框架设计
- PyTorch vs JAX vs TensorFlow设计哲学
- 自动微分实现原理
- **研究方向**: 编译器优化（XLA、TVM）、图优化

### Week 4-5: Transformer Architecture Deep Dive
**研究重点**: 架构创新与理论分析
- Attention的表达能力分析
- 位置编码的理论基础
- **研究方向**:
  - 状态空间模型（Mamba, RWKV）
  - 线性注意力
  - 无参数位置编码

### Week 6-7: Training at Scale
**研究重点**: 分布式训练系统
- 数据并行、模型并行、流水线并行
- ZeRO、FSDP、Megatron-LM
- **研究方向**:
  - 通信优化（NCCL、RDMA）
  - 混合并行策略自动搜索

### Week 8-9: Flash Attention & Memory Optimization
**研究重点**: 内存墙问题与I/O感知算法
- Roofline模型分析
- Tiling策略的理论最优性
- **研究方向**:
  - 稀疏注意力模式
  - 线性注意力
  - 分页注意力（PageAttention）

### Week 10-11: Scaling Laws & Emergence
**研究重点**: 规模定律的科学解释
- 幂律的物理基础
- Emergent abilities的争议
- **研究方向**:
  - 相变理论
  - 预测性扩展
  - 计算最优训练

### Week 12-13: Data Engineering
**研究重点**: 数据质量与效率
- 数据清洗的理论基础
- 去重的统计影响
- **研究方向**:
  - 数据混合策略
  - 课程学习
  - 数据归因

### Week 14-15: Pre-training & Continual Learning
**研究重点**: 预训练动力学
- 损失曲线分析
- 学习率调度理论
- **研究方向**:
  - 预训练后验证
  - 模型缝合（Model Soups）
  - 灾难性遗忘缓解

### Week 16-17: Alignment & RLHF
**研究重点**: 对齐理论与方法
- RLHF的理论基础
- DPO的几何解释
- **研究方向**:
  - Constitutional AI
  - RLAIF
  - 多目标优化

### Week 18-19: Inference & Deployment
**研究重点**: 推理系统与效率
- KV Cache优化理论
- 量化与剪枝
- **研究方向**:
  - 投机解码（Speculative Decoding）
  - 连续批处理
  - 服务级别目标（SLO）优化

---

## 🔬 研究方法论

### 如何读论文

1. **三遍阅读法**
   - 第一遍：标题→摘要→结论（5分钟）
   - 第二遍：图表→主要结果（30分钟）
   - 第三遍：完整阅读+复现（数小时）

2. **批判性阅读**
   - 作者想证明什么？
   - 假设是否合理？
   - 实验是否充分？
   - 有哪些局限性？

3. **做笔记模板**
   ```markdown
   ## 论文标题
   - 核心贡献：
   - 方法创新：
   - 实验结果：
   - 我的评价：
   - 相关论文：
   - 可复现性：
   ```

### 如何做研究

1. **选题策略**
   - 从课程作业延伸
   - 从论文limitations出发
   - 从工业界痛点切入

2. **实验设计**
   - 控制变量
   - 足够的数据点
   - 可复现性

3. **论文写作**
   - 故事线清晰
   - 实验充分
   - 代码开源

---

## 📚 核心论文清单（按主题）

### Tokenization
- [ ] Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)
- [ ] SentencePiece (Kudo & Richardson, 2018)
- [ ] BPE Dropout (Provilkov et al., 2020)
- [ ] Tokenization Impact Analysis (Mielke et al., 2021)

### Architecture
- [ ] Attention Is All You Need (Vaswani et al., 2017)
- [ ] Layer Normalization (Ba et al., 2016)
- [ ] Pre-LN Transformer (Xiong et al., 2020)
- [ ] RoPE (Su et al., 2021)
- [ ] ALiBi (Press et al., 2022)
- [ ] SwiGLU (Shazeer, 2020)

### Training
- [ ] AdamW (Loshchilov & Hutter, 2017)
- [ ] Warmup & Decay (Popel & Bojar, 2018)
- [ ] ZeRO (Rajbhandari et al., 2020)
- [ ] Megatron-LM (Shoeybi et al., 2019)
- [ ] FSDP (Zhao et al., 2023)

### Attention Optimization
- [ ] Flash Attention (Dao et al., 2022)
- [ ] Flash Attention-2 (Dao, 2023)
- [ ] PageAttention (Kwon et al., 2023)
- [ ] Linear Attention (Katharopoulos et al., 2020)

### Scaling Laws
- [ ] Scaling Laws (Kaplan et al., 2020)
- [ ] Chinchilla (Hoffmann et al., 2022)
- [ ] Emergent Abilities (Wei et al., 2022)
- [ ] Predictability (Hernandez et al., 2021)

### Data
- [ ] The Pile (Gao et al., 2020)
- [ ] Deduplication (Lee et al., 2022)
- [ ] Data Mixing (Xie et al., 2023)
- [ ] Data Quality (Touvron et al., 2023)

### Alignment
- [ ] RLHF (Ziegler et al., 2019)
- [ ] PPO (Schulman et al., 2017)
- [ ] DPO (Rafailov et al., 2023)
- [ ] Constitutional AI (Bai et al., 2022)

### Inference
- [ ] KV Cache (Pope et al., 2022)
- [ ] Quantization (Dettmers et al., 2022)
- [ ] Speculative Decoding (Leviathan et al., 2022)
- [ ] vLLM (Kwon et al., 2023)

---

## 🎯 研究项目建议

### 适合博士研究的扩展方向

1. **高效注意力机制**
   - 研究目标：亚二次复杂度注意力
   - 方法：核方法、稀疏模式、状态空间
   - 评估：理论复杂度+实际速度

2. **自适应训练策略**
   - 研究目标：动态调整训练配置
   - 方法：早停、数据选择、学习率自适应
   - 评估：收敛速度+最终性能

3. **多模态扩展**
   - 研究目标：统一的多模态架构
   - 方法：视觉编码器、跨模态对齐
   - 评估：多模态基准

4. **可信AI**
   - 研究目标：可解释性、安全性
   - 方法：归因分析、红队测试
   - 评估：人工评估+自动指标

---

## 📊 能力矩阵

完成本课程后，你应该具备：

| 能力 | 水平 |
|------|------|
| 实现Transformer | 专家 |
| 分布式训练 | 熟练 |
| 性能优化 | 熟练 |
| 数据处理 | 熟练 |
| RLHF | 熟练 |
| 系统设计 | 中级 |
| 论文阅读 | 熟练 |
| 实验设计 | 熟练 |

---

## 🔗 相关资源

### 课程资源
- 官网：https://cs336.stanford.edu/
- Piazza：课程论坛
- Ed：作业提交

### 社区
- Paper Reading Group
- 研究组会
- 行业会议（NeurIPS, ICML, ACL）

### 开源项目
- Megatron-LM
- DeepSpeed
- Colossal-AI
- vLLM

---

**祝你在LLM研究领域取得成功！** 🚀

