# Docker 配置

## 🐳 使用Docker运行CS336

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置工作目录
WORKDIR /workspace

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch==2.1.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install \
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
    jupyter \
    pytest

# 安装Flash Attention
RUN pip3 install ninja
RUN pip3 install flash-attn --no-build-isolation

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 暴露Jupyter端口
EXPOSE 8888

# 启动命令
CMD ["/bin/bash"]
```

### 构建镜像

```bash
docker build -t cs336:latest .
```

### 运行容器

```bash
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    cs336:latest
```

### Docker Compose

```yaml
version: '3.8'

services:
  cs336:
    build: .
    image: cs336:latest
    container_name: cs336
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./:/workspace
      - ./data:/data
    ports:
      - "8888:8888"
    stdin_open: true
    tty: true
    command: bash
```

---

*Docker配置 - CS336课程*

