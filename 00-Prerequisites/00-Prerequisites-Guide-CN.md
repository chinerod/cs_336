# CS336 前置知识学习指南

## 📚 为什么需要前置知识？

Stanford CS336 是一门**研究生级别**的课程，要求学生具备以下基础：

- ✅ **Python编程** - 熟练编写Python代码
- ✅ **PyTorch** - 熟悉深度学习框架
- ✅ **线性代数** - 矩阵运算、特征值等
- ✅ **微积分** - 梯度、偏导数、链式法则
- ✅ **概率论** - 概率分布、贝叶斯定理
- ✅ **机器学习基础** - 了解基本概念

⚠️ **如果没有这些基础，直接学CS336会非常痛苦！**

---

## 🎯 学习路径总览

```
完全零基础
    ↓ 2-4周
Python编程
    ↓ 2-3周
PyTorch基础
    ↓ 2-3周
数学基础（线代+微积分+概率）
    ↓ 2-3周
深度学习基础
    ↓ 1-2周
CS336预习（论文+概念）
    ↓
正式开始CS336！
```

**总时长：10-15周（每天2-3小时）**

---

## 📖 模块一：Python编程（2-4周）

### 如果你是完全零基础

**推荐资源**：
1. **Python官方教程**（英文）
   - https://docs.python.org/3/tutorial/
   - 读完前10章

2. **菜鸟教程Python**（中文）
   - https://www.runoob.com/python3/python3-tutorial.html
   - 适合中文用户

3. **CS50 Python**（推荐！）
   - Harvard的Python入门课
   - 已包含在我们的资料包中

### Python需要掌握什么？

```python
# 1. 基础语法
变量、数据类型、条件语句、循环
函数定义、类与对象

# 2. 数据结构
列表 list、字典 dict、元组 tuple、集合 set
列表推导式、生成器

# 3. 文件操作
读取文件、写入文件、CSV处理

# 4. 常用库
numpy - 数组操作
pandas - 数据处理
matplotlib - 绘图
```

### 练习项目
- 写一个计算器程序
- 处理CSV文件
- 实现一个简单的数据处理脚本

---

## 📖 模块二：PyTorch基础（2-3周）

### 如果你会Python但不会PyTorch

**推荐资源**：
1. **PyTorch官方60分钟入门**（英文）
   - https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

2. **PyTorch中文文档**（中文）
   - https://pytorch.apachecn.org/

3. **动手学深度学习（PyTorch版）**
   - 李沐的课程，有中文版本
   - https://zh.d2l.ai/

### PyTorch需要掌握什么？

```python
# 1. 张量操作
import torch
x = torch.tensor([1, 2, 3])
x = torch.zeros(2, 3)
x = torch.randn(2, 3)

# 2. GPU运算
x = x.cuda()  # 移到GPU
x = x.cpu()   # 移回CPU

# 3. 自动求导
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 梯度

# 4. 神经网络模块
import torch.nn as nn
linear = nn.Linear(10, 20)
relu = nn.ReLU()

# 5. 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.step()
optimizer.zero_grad()

# 6. 损失函数
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

### 练习项目
- 实现一个简单的神经网络
- 训练一个MNIST分类器
- 理解反向传播

---

## 📖 模块三：数学基础（2-3周）

### 3.1 线性代数（1周）

**需要掌握**：
1. **向量与矩阵**
   - 向量加法、数乘
   - 矩阵乘法
   - 转置、逆矩阵

2. **矩阵分解**
   - 特征值与特征向量
   - SVD（奇异值分解）

3. **常见运算**
   - 点积、外积
   - 范数（L1, L2）

**推荐资源**：
- **3Blue1Brown线性代数系列**（YouTube，有中文翻译）
  - 最直观的线性代数教程
  - https://www.bilibili.com/video/BV1ys411472E

- **MIT 18.06 线性代数**（英文）
  - Gilbert Strang的 legendary 课程

**快速复习**：
```python
import numpy as np

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # 或 np.dot(A, B)

# 转置
A_T = A.T

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD
U, S, Vh = np.linalg.svd(A)
```

---

### 3.2 微积分（1周）

**需要掌握**：
1. **导数**
   - 基本函数的导数
   - 链式法则（非常重要！）
   - 偏导数

2. **梯度**
   - 什么是梯度
   - 梯度下降
   - 方向导数

3. **优化基础**
   - 极大值/极小值
   - 凸函数

**推荐资源**：
- **3Blue1Brown微积分系列**（YouTube，有中文翻译）
  - https://www.bilibili.com/video/BV1qW411N7FU

- **MIT 18.01 单变量微积分**

**核心概念**：
```
# 链式法则（神经网络的核心！）
如果 z = f(y), y = g(x)
那么 dz/dx = dz/dy * dy/dx

# 偏导数
f(x, y) = x²y
∂f/∂x = 2xy
∂f/∂y = x²

# 梯度
∇f = [∂f/∂x, ∂f/∂y]
```

---

### 3.3 概率论与统计（1周）

**需要掌握**：
1. **基础概念**
   - 概率、条件概率
   - 贝叶斯定理
   - 期望、方差

2. **常见分布**
   - 均匀分布
   - 正态分布（高斯分布）
   - 伯努利分布

3. **最大似然估计**
   - 似然函数
   - 对数似然

**推荐资源**：
- **可汗学院概率统计**（英文，有字幕）
- **Stanford CS109**（课程资料）

**核心概念**：
```python
import numpy as np
from scipy import stats

# 高斯分布
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 1000)

# 贝叶斯定理
# P(A|B) = P(B|A) * P(A) / P(B)

# 最大似然估计
# 找到使 P(data|θ) 最大的 θ
```

---

## 📖 模块四：深度学习基础（2-3周）

### 如果你会PyTorch但不懂深度学习

**推荐资源**：
1. **CS231n（Stanford）**
   - 计算机视觉，但基础概念通用
   - http://cs231n.stanford.edu/

2. **神经网络与深度学习**
   - Michael Nielsen的在线书籍
   - http://neuralnetworksanddeeplearning.com/

3. **3Blue1Brown神经网络系列**
   - 最直观的神经网络讲解
   - https://www.bilibili.com/video/BV1bx411M7Zx

### 需要掌握的概念

```
1. 神经网络基础
   - 感知机
   - 激活函数（ReLU, Sigmoid, Tanh）
   - 前向传播、反向传播

2. 优化算法
   - 梯度下降
   - SGD
   - Adam

3. 正则化
   - Dropout
   - L2正则化
   - Batch Normalization

4. 卷积神经网络（CNN）基础
   - 卷积操作
   - 池化

5. 循环神经网络（RNN）基础
   - RNN、LSTM、GRU
```

### 练习项目
- 实现一个简单的全连接网络
- 训练一个图像分类器
- 理解反向传播手动计算

---

## 📖 模块五：CS336预习（1-2周）

### 在正式开始CS336之前

**1. 读关键论文**
- Attention Is All You Need（必读！）
- 读不懂没关系，先过一遍

**2. 看课程介绍**
- CS336官网的course description
- 了解课程大纲

**3. 准备环境**
- 按照我们的环境配置指南搭建
- 测试PyTorch是否能用GPU

**4. 加入社区**
- 找到学习伙伴
- 加入相关的学习群

---

## ⏰ 学习时间表示例

### 方案A：全职学习（每天4-6小时）

```
第1-3周：Python编程
第4-6周：PyTorch基础
第7-9周：数学基础
第10-12周：深度学习基础
第13-14周：CS336预习
第15周+：开始CS336
```

### 方案B：兼职学习（每天2小时）

```
第1-6周：Python编程
第7-10周：PyTorch基础
第11-16周：数学基础
第17-22周：深度学习基础
第23-24周：CS336预习
第25周+：开始CS336
```

---

## 🎯 快速自测

在开始CS336之前，检查你是否能完成：

- [ ] 用Python写一个简单的数据处理脚本
- [ ] 用PyTorch训练一个MNIST分类器
- [ ] 计算矩阵的特征值
- [ ] 求函数的导数和梯度
- [ ] 解释什么是贝叶斯定理
- [ ] 实现一个简单的前馈神经网络

**如果都能完成 → 可以开始CS336了！**

**如果有超过3项不会 → 先补前置知识**

---

## 📚 资源汇总

### 中文资源
- 菜鸟教程Python
- PyTorch中文文档
- 3Blue1Brown中文翻译（B站）
- 动手学深度学习

### 英文资源
- Python官方文档
- PyTorch官方教程
- Stanford CS231n
- MIT OpenCourseWare

### 视频平台
- Bilibili（中文教程）
- YouTube（英文教程）
- Coursera（系统课程）

---

## 💡 学习建议

1. **不要跳过前置知识** - 地基不牢，地动山摇
2. **边学边练** - 只看视频不练习等于没学
3. **不懂就问** - 社区、论坛、AI助手都是好帮手
4. **循序渐进** - 不要试图一周学完所有
5. **保持耐心** - 打好基础后面会轻松很多

---

## ❓ 常见问题

**Q: 我数学不好，能学CS336吗？**
A: 可以先学，但会很吃力。建议至少学会微积分和线性代数基础。

**Q: 没有GPU能学吗？**
A: CS336必须要有GPU， Assignment 2开始必须用GPU。可以用云端GPU（Lambda/Vast等）。

**Q: 英语不好怎么办？**
A: 课程是英文的，但可以配合中文资料。建议同时提升英语。

**Q: 要多久才能开始CS336？**
A: 有基础：可以直接开始。零基础：10-15周前置学习。

---

**记住：慢就是快，打好基础最重要！** 💪

**现在开始你的前置学习之旅吧！** 🚀
