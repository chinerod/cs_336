# 系统论文精选

## 📚 训练系统

### 1. Megatron-LM

**"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (Shoeybi et al., 2019)

**核心贡献**：
- 张量并行
- Pipeline并行
- 高效通信

**架构**：
```
层内并行：列并行 + 行并行
层间并行：Pipeline
```

---

### 2. ZeRO

**"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (Rajbhandari et al., 2020)

**创新点**：
- ZeRO-1: 优化器状态分片
- ZeRO-2: 梯度分片
- ZeRO-3: 参数分片

**内存节省**：
- ZeRO-1: 4x
- ZeRO-2: 8x
- ZeRO-3: 与数据并行度线性相关

---

### 3. FSDP

**"Fully Sharded Data Parallel: faster AI training with fewer GPUs"** (Zhao et al., 2023)

**特点**：
- PyTorch原生
- ZeRO-3类似
- 更易用

---

### 4. DeepSpeed

**"DeepSpeed: Extreme-Scale Model Training for Everyone"**

**功能**：
- ZeRO优化
- 1-bit Adam
- Curriculum Learning
- MoE训练

---

## 📚 推理系统

### 1. vLLM

**"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023)

**核心**：
- PagedAttention
- 非连续KV Cache
- 动态内存分配

**性能**：
- 2-4x吞吐量提升

---

### 2. Orca

**"Orca: A Distributed Serving System for Transformer-Based Generative Models"**

**创新**：
- 迭代级调度
- Selective batching
- 并行执行

---

### 3. TensorRT-LLM

**NVIDIA推理优化**

**优化**：
- Kernel融合
- FP8量化
- 多GPU并行

---

## 📚 编译优化

### 1. XLA

**TensorFlow编译器**

**功能**：
- 图优化
- 自动微分
- 多后端支持

### 2. TVM

**"TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"**

**特点**：
- 自动算子优化
- 多硬件支持
- 学习优化

### 3. Triton

**OpenAI GPU编程**

**优势**：
- Python语法
- 自动优化
- 高性能

---

## 🎯 研究方向

### 当前热点

1. **长上下文支持**
   - 高效注意力
   - 上下文压缩

2. **边缘设备部署**
   - 模型压缩
   - 量化推理

3. **多模态系统**
   - 统一架构
   - 高效调度

4. **AutoML for Systems**
   - 自动并行策略
   - 自动内存管理

---

*系统论文清单 - 持续更新*

