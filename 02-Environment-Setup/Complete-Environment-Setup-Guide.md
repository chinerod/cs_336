# CS336 Complete Environment Setup Guide
# Stanford CS336: Language Modeling from Scratch

## System Requirements

### Minimum Requirements
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- RAM: 16GB system memory
- Storage: 100GB free space
- OS: Linux (Ubuntu 20.04+ recommended) or macOS

### Recommended Configuration
- GPU: NVIDIA RTX 3090 (24GB) or A100 (40GB)
- RAM: 32GB+ system memory
- Storage: 500GB NVMe SSD
- CPU: 8+ cores
- Network: Stable internet for downloading datasets

## Option 1: Local Setup (Linux)

### Step 1: Install NVIDIA Drivers
```bash
# Check if NVIDIA GPU is present
lspci | grep -i nvidia

# Install drivers (Ubuntu)
sudo apt update
sudo apt install -y nvidia-driver-525
sudo reboot

# Verify
nvidia-smi
```

### Step 2: Install CUDA Toolkit
```bash
# Download CUDA 11.8 from NVIDIA website
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# Install
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Step 3: Install Python Environment
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create CS336 environment
conda create -n cs336 python=3.10 -y
conda activate cs336
```

### Step 4: Install PyTorch with CUDA
```bash
# CUDA 11.8 version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Step 5: Install Core Dependencies
```bash
# Essential packages
pip install einops triton transformers datasets accelerate

# Development tools
pip install wandb tensorboard jupyter ipython

# Scientific computing
pip install numpy scipy scikit-learn matplotlib seaborn pandas

# Code quality
pip install black isort flake8 mypy pytest

# Profiling tools (for Assignment 2)
pip install py-spy line_profiler memory_profiler
```

### Step 6: Install Flash Attention
```bash
# Install ninja for faster compilation
pip install ninja

# Set compilation jobs
export MAX_JOBS=4

# Install flash-attention
pip install flash-attn --no-build-isolation

# Verify
python -c "import flash_attn; print(flash_attn.__version__)"
```

### Step 7: Additional Tools
```bash
# Install Triton (already included with PyTorch 2.0+)
pip install triton

# Install einops for tensor operations
pip install einops

# Install huggingface datasets
pip install datasets

# Install accelerate for distributed training
pip install accelerate
```

## Option 2: Cloud GPU Setup

### Lambda Cloud (Recommended)

1. **Create Instance**
   - GPU: 1x A10 (24GB) or 1x A100 (40GB)
   - Image: PyTorch 2.0 (Ubuntu 22.04)
   - Region: US West 2 or Asia Pacific

2. **Connect via SSH**
   ```bash
   ssh -i lambda_key.pem ubuntu@<instance-ip>
   ```

3. **Environment Already Pre-configured**
   ```bash
   # Just install additional packages
   pip install einops triton flash-attn transformers datasets accelerate wandb
   ```

### Vast.ai Setup

1. **Find Instance**
   - Filter: RTX 3090 or better
   - Reliability Score: >95%
   - Image: PyTorch

2. **Connect**
   ```bash
   ssh -p <port> root@<ip> -i ~/.ssh/id_rsa
   ```

3. **Setup Environment**
   ```bash
   apt-get update && apt-get install -y git vim htop build-essential

   # Create conda environment
   conda create -n cs336 python=3.10 -y
   conda activate cs336

   # Install PyTorch
   pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

   # Install dependencies
   pip install einops triton transformers datasets accelerate
   ```

## Option 3: Docker Setup

### Dockerfile
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    einops \
    triton \
    transformers \
    datasets \
    accelerate \
    wandb \
    numpy \
    scipy \
    matplotlib \
    pandas \
    jupyter \
    flash-attn

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

CMD ["/bin/bash"]
```

### Build and Run
```bash
# Build image
docker build -t cs336-env .

# Run container
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    cs336-env

# Inside container
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Verification Script

Create `verify_setup.py`:

```python
#!/usr/bin/env python3
"""Verify CS336 environment setup"""
import sys
import subprocess

def check_python():
    print("✓ Python version:", sys.version.split()[0])
    assert sys.version_info >= (3, 10), "Python 3.10+ required"

def check_pytorch():
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def check_packages():
    packages = [
        'einops', 'triton', 'transformers',
        'datasets', 'accelerate', 'numpy', 'scipy'
    ]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} NOT installed")

def check_flash_attn():
    try:
        import flash_attn
        print(f"✓ Flash Attention: {flash_attn.__version__}")
    except ImportError:
        print("✗ Flash Attention NOT installed (optional for Assignment 2)")

def test_gpu():
    import torch
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        print("✓ GPU computation test passed")

def main():
    print("="*60)
    print("CS336 Environment Verification")
    print("="*60)

    check_python()
    check_pytorch()
    check_packages()
    check_flash_attn()
    test_gpu()

    print("="*60)
    print("Setup verification complete!")
    print("="*60)

if __name__ == "__main__":
    main()
```

Run verification:
```bash
python verify_setup.py
```

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```python
# Solution 1: Reduce batch size
batch_size = 4  # instead of 8

# Solution 2: Use gradient accumulation
accumulation_steps = 4
loss = loss / accumulation_steps
loss.backward()

# Solution 3: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 4: Use mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input)
```

### Issue 2: Flash Attention Installation Fails
```bash
# Install build dependencies
sudo apt-get install build-essential

# Set environment variables
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# Retry installation
pip install flash-attn --no-build-isolation
```

### Issue 3: Triton Compilation Error
```bash
# Update Triton
pip install --upgrade triton

# Or install from source
pip install git+https://github.com/openai/triton.git
```

### Issue 4: Slow Data Loading
```python
# In DataLoader, increase num_workers
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Increase this
    pin_memory=True,
    prefetch_factor=2,
)
```

## Performance Optimization Tips

### 1. Enable TF32 (for Ampere GPUs)
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Automatic Mixed Precision
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Compile Model (PyTorch 2.0+)
```python
# Compile for faster training
model = torch.compile(model)
```

## Assignment-Specific Requirements

### Assignment 1 (Basics)
- Requirements: 8GB+ VRAM
- Estimated time: 10-20 hours
- Cost (cloud): $5-15

### Assignment 2 (Systems)
- Requirements: 16GB+ VRAM, Triton
- Estimated time: 20-30 hours
- Cost (cloud): $15-30

### Assignment 3 (Scaling)
- Requirements: 24GB+ VRAM
- Estimated time: 15-25 hours
- Cost (cloud): $15-25

### Assignment 4 (Data)
- Requirements: 16GB+ VRAM, 200GB storage
- Estimated time: 20-30 hours
- Cost (cloud): $20-40

### Assignment 5 (Alignment)
- Requirements: 24GB+ VRAM
- Estimated time: 20-30 hours
- Cost (cloud): $20-35

## Cloud GPU Cost Estimation

| Provider | GPU | VRAM | Price/Hour | Weekly Cost (20h) |
|----------|-----|------|------------|-------------------|
| Lambda | A10 | 24GB | $0.60 | $12 |
| Lambda | A100 | 40GB | $1.10 | $22 |
| Vast.ai | RTX 3090 | 24GB | $0.40 | $8 |
| Vast.ai | RTX 4090 | 24GB | $0.60 | $12 |
| AutoDL | RTX 3090 | 24GB | ¥2.5 | ~$7 |

Total estimated cost for entire course: **$100-300**

## Getting Help

- Course Slack: cs336-spr2026
- Ed Discussion: Check course website
- GitHub Issues: stanford-cs336 repositories
- Office Hours: See course calendar

## Next Steps

1. ✓ Complete environment setup
2. ✓ Run verification script
3. ✓ Clone Assignment 1 repository
4. ✓ Start with Week 1 materials
5. ✓ Join course Slack community

---

**Good luck with CS336!** 🚀
