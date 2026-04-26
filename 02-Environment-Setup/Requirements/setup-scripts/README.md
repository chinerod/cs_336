# 环境配置脚本

## 🚀 自动安装脚本

### 安装脚本

```bash
#!/bin/bash
# setup.sh - CS336环境自动安装脚本

echo "================================"
echo "CS336 环境配置脚本"
echo "================================"

# 检查CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: NVIDIA驱动未安装"
    exit 1
fi

echo "CUDA版本:"
nvidia-smi

# 创建conda环境
echo "创建conda环境..."
conda create -n cs336 python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cs336

# 安装PyTorch
echo "安装PyTorch..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
echo "安装核心依赖..."
pip install \
    einops \
    triton \
    transformers \
    datasets \
    accelerate \
    wandb \
    sentencepiece \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    tqdm \
    pytest \
    black \
    pylint \
    mypy

# 安装Flash Attention
echo "安装Flash Attention..."
pip install ninja
export MAX_JOBS=4
pip install flash-attn --no-build-isolation

# 验证安装
echo "验证安装..."
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "================================"
echo "安装完成!"
echo "请运行: conda activate cs336"
echo "================================"
```

### 使用方式

```bash
chmod +x setup.sh
./setup.sh
```

---

## 🔧 配置文件

### requirements.txt

```
torch==2.1.0
torchvision
torchaudio
einops
triton
transformers
datasets
accelerate
wandb
sentencepiece
numpy
pandas
matplotlib
seaborn
jupyter
ipykernel
tqdm
pytest
black
pylint
mypy
```

### environment.yml

```yaml
name: cs336
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.0
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - pip
  - pip:
    - einops
    - triton
    - transformers
    - datasets
    - accelerate
    - wandb
    - sentencepiece
    - flash-attn
```

### 创建环境

```bash
conda env create -f environment.yml
conda activate cs336
```

---

*环境配置脚本 - CS336课程*

