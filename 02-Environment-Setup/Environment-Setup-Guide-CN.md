# CS336 环境配置完全指南（中文版）

## 🎯 前言

配置环境是学CS336的第一步，也是最让人头疼的一步。

这份指南会手把手教你配置，**保证你一次成功！**

---

## 💻 方案选择

| 方案 | 适合人群 | 成本 | 难度 | 推荐度 |
|------|---------|------|------|--------|
| **云端GPU** | 所有人 | ¥500-1500 | ⭐ | ⭐⭐⭐⭐⭐ |
| **本地Linux** | 有Linux基础 | ¥0（有显卡） | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **本地Windows** | 只熟悉Windows | ¥0（有显卡） | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Mac** | 只有Mac | 无法本地训练 | - | ⭐ |

**推荐：云端GPU**（最简单，不需要折腾）

---

## ☁️ 方案一：云端GPU（推荐）

### 为什么推荐云端？

✅ **不需要买显卡** - 租用最便宜的¥600就能学完
✅ **配置简单** - 预装好了PyTorch
✅ **随时随地** - 有网就能学
✅ **弹性付费** - 用多少付多少

### 平台选择

| 平台 | 价格 | 优点 | 缺点 |
|------|------|------|------|
| **Lambda** | $0.6/小时 | 预装环境，最省心 | 经常没货 |
| **Vast** | $0.4/小时 | 最便宜 | 要自己配 |
| **AutoDL** | ¥2.5/小时 | 国内访问快 | 略贵 |

**新手推荐：Lambda**

---

### Lambda配置教程（详细步骤）

#### 第一步：注册账号

1. 打开 https://lambdalabs.com/
2. 点击 "Sign Up"
3. 用邮箱注册
4. 验证邮箱

#### 第二步：添加付款方式

1. 登录后点击右上角头像
2. 选择 "Billing"
3. 添加信用卡或PayPal
4. 等待审核（通常几分钟）

#### 第三步：创建实例

1. 点击 "Launch Instance"
2. 选择配置：
   ```
   GPU: 1x GPU (A10推荐)
   Region: US West 2 (California)
   Image: PyTorch 2.0 (Ubuntu 22.04)
   ```
3. 点击 "Launch"
4. 等待启动（约2-3分钟）

#### 第四步：连接服务器

**方法A：浏览器直接打开（最简单）**

1. 在Dashboard找到你的实例
2. 点击 "Launch Jupyter"
3. 直接在浏览器中写代码

**方法B：SSH连接（推荐）**

1. 下载私钥（创建时会提示下载）
2. 打开终端（Windows用PowerShell）
3. 运行：
   ```bash
   chmod 600 lambda_key.pem
   ssh -i lambda_key.pem ubuntu@你的IP地址
   ```

#### 第五步：安装CS336依赖

连接成功后，运行：

```bash
# 升级pip
pip install --upgrade pip

# 安装核心依赖
pip install einops triton transformers datasets accelerate wandb

# 安装Flash Attention（Assignment 2需要）
pip install ninja
export MAX_JOBS=4
pip install flash-attn --no-build-isolation

# 验证安装
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

看到GPU名称就成功了！

#### 第六步：下载课程代码

```bash
# 创建一个文件夹
mkdir cs336
cd cs336

# 下载Assignment代码（从GitHub）
git clone https://github.com/stanford-cs336/assignment1-basics.git

# 进入查看
ls assignment1-basics/
```

#### 第七步：开始写代码

**如果用Jupyter：**
1. 浏览器打开Jupyter
2. 新建Notebook
3. 直接写代码

**如果用SSH+VS Code：**
1. 安装VS Code
2. 安装Remote-SSH插件
3. 配置SSH连接
4. 像本地一样写代码

---

### Vast配置教程（最便宜）

#### 第一步：注册
1. 打开 https://vast.ai/
2. 注册账号
3. 添加付款方式

#### 第二步：找机器
1. 点击 "Create Instance"
2. 筛选：
   ```
   GPU: RTX 3090
   Image: PyTorch
   Reliability: >95%
   ```
3. 选择最便宜的
4. 点击 "Rent"

#### 第三步：连接
1. 创建后会显示SSH命令
2. 复制到终端运行
3. 密码会显示在页面上

#### 第四步：配置环境

```bash
# Vast机器比较干净，要装的东西多

# 1. 安装基本工具
apt-get update
apt-get install -y git vim htop

# 2. 安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 3. 重启shell，然后创建环境
conda create -n cs336 python=3.10 -y
conda activate cs336

# 4. 安装PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 5. 安装其他依赖
pip install einops triton transformers datasets accelerate wandb
```

---

### AutoDL配置教程（国内用户）

#### 第一步：注册
1. 打开 https://www.autodl.com/
2. 注册账号
3. 实名认证（需要身份证）

#### 第二步：创建实例
1. 点击 "租用新实例"
2. 选择地区（选离你近的）
3. 选择GPU（RTX 3090）
4. 选择镜像（PyTorch 2.0）
5. 点击 "立即创建"

#### 第三步：连接
AutoDL提供多种连接方式：
- JupyterLab（浏览器）
- SSH
- VS Code远程

推荐使用 **JupyterLab**，开箱即用。

---

## 💰 成本控制攻略

### 省钱技巧

1. **及时关机**
   - 不用的时候立刻关机
   - 按小时计费，关机就不花钱

2. **用竞价实例**（Vast）
   - 价格便宜50%
   - 可能被抢占，但作业可以保存checkpoint

3. **合理安排时间**
   - 一次训练2-3小时
   - 中间休息就关机

4. **Assignment 1-3用便宜配置**
   - 3090够用
   - Assignment 4-5再用A100

### 预算参考

| 方案 | 每周学习 | 10周总成本 |
|------|---------|------------|
| **省钱版** | Vast 3090, 20小时/周 | ¥600-800 |
| **标准版** | Lambda A10, 20小时/周 | ¥1000-1200 |
| **舒适版** | Lambda A100, 不限时 | ¥2000-3000 |

---

## 🐧 方案二：本地Linux配置

### 适用人群
- 已经有NVIDIA显卡（8G+显存）
- 会用Linux

### 配置步骤

#### 第一步：检查显卡
```bash
# 查看显卡型号
nvidia-smi

# 应该输出类似：
# NVIDIA-SMI 525.85.12    Driver Version: 525.85.12
```

#### 第二步：安装CUDA
```bash
# 下载CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

#### 第三步：安装Conda
```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### 第四步：创建环境并安装依赖
```bash
conda create -n cs336 python=3.10 -y
conda activate cs336

# 安装PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install einops triton transformers datasets accelerate wandb
```

---

## 🪟 方案三：Windows配置

### 不推荐，但如果必须用...

Windows配置很麻烦，建议：
1. **装WSL2**（Windows Subsystem for Linux）
2. 在WSL2里按照Linux步骤配置

### WSL2安装步骤

1. **开启WSL2**
   ```powershell
   # 以管理员身份运行PowerShell
   wsl --install
   ```

2. **重启电脑**

3. **安装Ubuntu**
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

4. **在WSL2里配置**
   - 进入WSL2
   - 按照"本地Linux配置"步骤

---

## 🍎 方案四：Mac配置

### 坏消息
**Mac不能本地训练大模型** - 因为没有NVIDIA显卡

### 好消息
**可以用云端GPU！**

Mac用户直接看"方案一：云端GPU"

---

## ✅ 环境验证

配置完成后，运行以下代码验证：

```bash
python -c "
import torch
import sys

print(f'Python版本: {sys.version}')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = x @ x.T
    print('✅ GPU计算测试通过！')
else:
    print('❌ CUDA不可用')
"
```

看到"✅ GPU计算测试通过"就成功了！

---

## ❌ 常见问题

### 问题1：CUDA版本不匹配
**现象**：`CUDA error: no kernel image`

**解决**：
```bash
# 查看CUDA版本
nvidia-smi

# 安装对应PyTorch版本
# CUDA 11.8
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 问题2：显存不足
**现象**：`CUDA out of memory`

**解决**：
```python
# 减小batch size
batch_size = 4  # 原来是8

# 使用梯度累积
accumulation_steps = 4

# 清空缓存
import torch
torch.cuda.empty_cache()
```

### 问题3：Flash Attention装不上
**现象**：编译错误

**解决**：
```bash
# 安装编译工具
sudo apt-get install build-essential

# 设置编译参数
export MAX_JOBS=4
pip install flash-attn --no-build-isolation
```

### 问题4：下载速度慢
**解决**：
```bash
# 用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch

# 或者用代理
export HTTPS_PROXY=http://your-proxy:port
```

---

## 📞 需要帮助？

如果配置遇到问题：

1. **看报错信息** - 复制到Google搜索
2. **问AI助手** - ChatGPT/Claude可以帮忙
3. **查官方文档** - PyTorch/CUDA官网
4. **加群问** - 买VIP版进答疑群

---

## 🎉 恭喜！

配置完成，现在可以开始学CS336了！

**下一步：** 阅读「学习计划」，开始Week 1！

---

**祝你学习顺利！** 🚀
