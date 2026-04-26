# CS336 大模型课程完整学习资料包
# Stanford CS336: Language Modeling from Scratch

![CS336](https://img.shields.io/badge/CS336-Stanford-red)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![中文](https://img.shields.io/badge/语言-中文-brightgreen)

---

## 📦 资料包简介

这是目前**市面上最完整**的 CS336 "从零构建语言模型" 中文学习资料包。

### 💡 适合谁？

✅ **零基础想转大模型** - 从前置知识到课程完整指南
✅ **在职工程师** - 系统学习LLM底层原理
✅ **在校学生** - 辅助完成CS336作业
✅ **自学者** - 无需上课也能学会手撸大模型

---

## 📦 包含内容

### 📚 00-前置知识（零基础必备）
- ✅ **Python编程入门** - 从0开始学习Python
- ✅ **PyTorch快速上手** - 深度学习框架教程
- ✅ **数学基础速成** - 线性代数/微积分/概率论
- ✅ **深度学习基础** - 神经网络概念详解
- ✅ **10-15周学习计划** - 零基础到CS336

### 📘 01-课程概览
- ✅ 课程大纲（中文翻译）
- ✅ 学习路径建议
- ✅ 先修知识检查清单

### 📗 02-环境配置（超详细）
- ✅ **本地配置** - Windows/Mac/Linux
- ✅ **云端GPU** - Lambda/Vast/AutoDL详细教程
- ✅ **Docker一键部署**
- ✅ **常见问题排查**

### 📓 03-讲义资料
- ✅ Week 0-10 每周学习重点
- ✅ PyTorch + einops 教程
- ✅ Triton CUDA编程指南

### 📔 04-作业参考
- ✅ 5个Assignment说明
- ✅ 解题思路分析
- ✅ 参考答案（加密）

### 📙 05-必读论文（24篇）
- ✅ **核心论文** - Attention/GPT/Flash Attention
- ✅ **Scaling Laws** - OpenAI和DeepMind
- ✅ **RLHF** - InstructGPT/DPO
- ✅ **数据处理** - 去重/过滤/混合
- ✅ **自动下载脚本**

### 📒 06-扩展阅读
- ✅ LLaMA/PaLM/Mistral架构
- ✅ 系统优化论文

### 💻 07-代码参考库
- ✅ **vanilla-transformer.py** - 完整Transformer（中文注释）
- ✅ **flash-attention-reference.py** - Flash Attention实现
- ✅ **training-utilities.py** - 训练工具
- ✅ 全部中文注释

### 📺 08-视频资源
- ✅ 课程视频链接
- ✅ 中文翻译资源

### 📋 09-练习题
- ✅ 补充习题
- ✅ 面试题集锦

### 📄 10-速查表
- ✅ **PyTorch中文速查表**
- ✅ Triton/CUDA速查
- ✅ Git/Linux命令

### 📅 11-学习计划
- ✅ **10周速成**（全职）
- ✅ **15周标准**（兼职）
- ✅ **20周轻松**（自学）
- ✅ 每日学习模板


---

## 🚀 快速开始

### 如果你是零基础

**第一步：学习前置知识（10-15周）**
```
阅读：00-Prerequisites/00-前置知识学习指南.md
按顺序学习：Python → PyTorch → 数学 → 深度学习
```

**第二步：配置环境**
```
阅读：02-Environment-Setup/完整环境配置指南.md
选择：本地配置 or 云端GPU
```

**第三步：开始学习**
```
阅读：11-Study-Plans/学习计划.md
选择适合你的学习路径
```

### 如果你有基础

**直接跳到第三步！**

---

## 📊 课程地图

| 周次 | 主题 | 作业 | 核心论文 | 代码 |
|------|------|------|----------|------|
| 0-1 | Tokenization | A1开始 | Tokenization论文 | Tokenizer实现 |
| 2 | PyTorch/einops | A1继续 | Roofline模型 | 性能分析 |
| 3-5 | Transformer | A1截止 | **Attention论文** | 完整Transformer |
| 6 | GPU/TPU | A2开始 | **Flash Attention** | Triton kernel |
| 7 | 并行计算 | A2 | Megatron/ZeRO | 分布式训练 |
| 8-10 | Scaling Laws | A2/A3 | **Scaling论文** | Scaling实验 |
| 11-14 | 数据处理 | A4 | 去重/过滤 | 数据管道 |
| 15+ | 对齐 | A5 | **RLHF/DPO** | RL训练 |

---

## 💰 投入成本

### 时间成本
- **有基础**：10周，每周12-15小时
- **零基础**：20-25周（含前置学习）




---

## 📚 核心论文（24篇）

### 必读（Priority 1）
1. **Attention Is All You Need** - Transformer奠基
2. **BERT** - 双向预训练
3. **GPT-3** - 大规模语言模型
4. **Flash Attention** - 高效Attention
5. **InstructGPT** - RLHF原论文
6. **Scaling Laws** - OpenAI

### 重要（Priority 2）
- Flash Attention 2
- Chinchilla
- ZeRO
- DPO
- 数据处理相关

全部论文含**中文阅读指南**和**重点标注**。

---

## 💻 代码库

### Transformer实现
```python
# 完整Transformer（中文注释）
class TransformerLM(nn.Module):
    """
    Transformer语言模型
    参考：Attention Is All You Need
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8):
        # 详细注释...
```

### Flash Attention
```python
# Flash Attention V1/V2实现
class FlashAttentionV1:
    """
    Flash Attention算法
    大幅减少显存占用
    """
    # 详细实现...
```

---


---

## ⚠️ 注意事项

### 学术诚信
- 本资料包仅供**学习参考**
- 请遵守Stanford Honor Code
- **不要直接抄袭**Assignment答案
- 代码用于理解原理，不是提交答案


---


