# CUDA 速查表

## 📚 基础概念

### GPU架构
- SM (Streaming Multiprocessor)
- Warp (32 threads)
- Thread Block
- Grid

### 内存层次
```
Register (最快, 每个线程)
  ↓
Shared Memory (快, block内共享)
  ↓
Global Memory (慢, 所有线程)
```

---

## 📚 核函数

### 定义
```cuda
__global__ void kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}
```

### 启动
```cuda
kernel<<<(n+255)/256, 256>>>(d_out, d_in, n);
```

---

## 📚 内存管理

### 分配/释放
```cuda
cudaMalloc(&d_ptr, size);
cudaFree(d_ptr);

// 拷贝
cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
```

---

## 📚 PyTorch CUDA

```python
x = torch.randn(1000, device='cuda')
torch.cuda.synchronize()
```

---

*CUDA速查表*
