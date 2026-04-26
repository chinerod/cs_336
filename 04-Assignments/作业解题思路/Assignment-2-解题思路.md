# Assignment 2 解题思路指南

## 📝 作业概述

**主题**: Flash Attention实现与优化
**目标**: 手写Flash Attention CUDA kernel，理解内存优化
**难度**: ⭐⭐⭐⭐⭐（最难的作业）
**预计时间**: 2-3周
**前置要求**: 理解GPU架构、Triton基础

---

## 📚 知识准备

### 必须理解的概念

#### 1. GPU内存层次结构
```
HBM (High Bandwidth Memory)
    ↓ 访问慢 (~1TB/s), 容量大 (16GB+)
SRAM (Shared Memory / L1 Cache)
    ↓ 访问快 (~10TB/s), 容量小 (128KB per SM)
寄存器
    ↓ 最快, 容量最小
```

**为什么Flash Attention快？**
- 标准Attention: O(N²)的HBM读写
- Flash Attention: 通过tiling减少HBM访问，主要用SRAM

#### 2. Tiling算法
- 将大矩阵分成小块（tiles）
- 在SRAM中计算小块的结果
- 增量更新输出

#### 3. Online Softmax
- 不存储完整的attention matrix
- 增量计算softmax
- 需要跟踪running maximum和running sum

---

## 🗂️ 作业结构

### Part 1: 标准Attention实现（Baseline）

首先实现标准attention作为baseline：

```python
def standard_attention(Q, K, V, causal=False):
    """
    Q, K, V: [batch, n_heads, seq_len, d_head]
    """
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

    if causal:
        mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output
```

**复杂度分析**:
- 计算: O(N²d)
- HBM访问: O(N²) - 需要存储N×N的attention matrix
- 当N很大时，内存是瓶颈

---

### Part 2: Flash Attention核心算法

#### 2.1 算法理解

Flash Attention的核心思想：
1. 将Q, K, V分成blocks
2. 每次加载一个Q block和K/V blocks到SRAM
3. 在SRAM中计算部分attention
4. 增量更新输出，不存储中间矩阵

#### 2.2 Tiling策略

```python
# 假设：
# - seq_len = N
# - block_size = Bc (for K, V) / Br (for Q)
# - 需要: Bc * d, Br * d 能放进SRAM

# 外层循环：遍历Q的blocks (Tr = ceil(N/Br))
for i in range(Tr):
    Qi = Q[i*Br:(i+1)*Br, :]  # Load Q block to SRAM

    # 初始化累加器
    mi = torch.zeros(Br, 1)  # running max
    li = torch.zeros(Br, 1)  # running sum exp
    Oi = torch.zeros(Br, d)  # output accumulator

    # 内层循环：遍历K, V的blocks (Tc = ceil(N/Bc))
    for j in range(Tc):
        Kj = K[j*Bc:(j+1)*Bc, :]  # Load K block
        Vj = V[j*Bc:(j+1)*Bc, :]  # Load V block

        # 在SRAM中计算Sij = Qi @ Kj^T
        Sij = Qi @ Kj.T

        # Online softmax更新
        mij = torch.max(Sij, dim=1, keepdim=True).values
        Pij = torch.exp(Sij - mij)
        lij = torch.sum(Pij, dim=1, keepdim=True)

        # 更新running statistics
        mi_new = torch.max(mi, mij)
        li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij

        # 更新输出
        Oi = (li * torch.exp(mi - mi_new) * Oi + torch.exp(mij - mi_new) * (Pij @ Vj)) / li_new

        # 更新running stats
        mi = mi_new
        li = li_new
```

#### 2.3 Online Softmax详解

**为什么需要online softmax？**

标准softmax:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

但如果我们分块计算，不能直接softmax每个块再组合。

**增量计算方法**:

```python
# 假设我们已经有前k个数的softmax相关信息
m_prev = max(x_1, ..., x_k)  # 之前的最大值
l_prev = sum(exp(x_i - m_prev))  # 之前的归一化项

# 现在来了新的数 x_{k+1}
m_new = max(m_prev, x_{k+1})
l_new = exp(m_prev - m_new) * l_prev + exp(x_{k+1} - m_new)

# 这样我们可以增量更新，不需要存储所有exp值
```

#### 2.4 完整Flash Attention Forward

```python
def flash_attention_forward(Q, K, V, causal=False, block_size=64):
    """
    Q, K, V: [batch, n_heads, seq_len, d_head]
    """
    batch, n_heads, N, d = Q.shape

    # 输出
    O = torch.zeros_like(Q)

    # 分块参数
    Br = block_size  # Q的block大小
    Bc = block_size  # K, V的block大小

    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    for b in range(batch):
        for h in range(n_heads):
            Q_bh = Q[b, h]  # [N, d]
            K_bh = K[b, h]
            V_bh = V[b, h]

            for i in range(Tr):
                # 加载Q block
                Qi = Q_bh[i*Br:(i+1)*Br]  # [Br, d]
                Oi = torch.zeros(Br, d)
                mi = torch.full((Br, 1), float('-inf'))
                li = torch.zeros(Br, 1)

                # 确定K, V的范围（causal mask）
                j_end = Tc if not causal else min(Tc, (i+1) * Br // Bc)

                for j in range(j_end):
                    # 加载K, V block
                    Kj = K_bh[j*Bc:(j+1)*Bc]  # [Bc, d]
                    Vj = V_bh[j*Bc:(j+1)*Bc]  # [Bc, d]

                    # 计算S = Qi @ Kj^T / sqrt(d)
                    Sij = Qi @ Kj.T / math.sqrt(d)  # [Br, Bc]

                    # Online softmax
                    mij = torch.max(Sij, dim=1, keepdim=True)[0]
                    Pij = torch.exp(Sij - mij)
                    lij = torch.sum(Pij, dim=1, keepdim=True)

                    # 更新running statistics
                    mi_new = torch.maximum(mi, mij)
                    li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij

                    # 更新输出
                    # O_new = (old_term + new_term) / new_normalization
                    Oi = (li * torch.exp(mi - mi_new) * Oi +
                          torch.exp(mij - mi_new) * (Pij @ Vj)) / li_new

                    mi = mi_new
                    li = li_new

                O[b, h, i*Br:(i+1)*Br] = Oi

    return O
```

---

### Part 3: Triton Kernel实现

使用Triton写CUDA kernel：

```python
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH, N_HEADS, SEQ_LEN, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 程序ID
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_block = tl.program_id(2)  # Q的block id

    # 计算偏移
    q_offs = batch_id * stride_qb + head_id * stride_qh
    k_offs = batch_id * stride_kb + head_id * stride_kh
    v_offs = batch_id * stride_vb + head_id * stride_vh
    o_offs = batch_id * stride_ob + head_id * stride_oh

    # 加载Q block
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + q_offs + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_LEN, other=0.0)

    # 初始化累加器
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 遍历K, V blocks
    for n_block in range(0, (SEQ_LEN + BLOCK_N - 1) // BLOCK_N):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

        # 加载K block
        k_ptrs = K + k_offs + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

        # 计算Q @ K^T
        qk = tl.dot(q, tl.trans(k))
        qk = qk * (1.0 / tl.sqrt(float(HEAD_DIM)))

        # Causal mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax
        m_new = tl.maximum(m, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        l_new = tl.exp(m - m_new) * l + tl.sum(p, axis=1)

        # 加载V block
        v_ptrs = V + v_offs + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < SEQ_LEN, other=0.0)

        # 更新输出
        acc = acc * (l[:, None] * tl.exp(m - m_new) / l_new)[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v) / l_new[:, None]

        m = m_new
        l = l_new

    # 存储输出
    o_ptrs = Out + o_offs + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < SEQ_LEN)
```

---

### Part 4: Backward实现（可选/进阶）

Flash Attention的backward更复杂，需要重新计算forward的中间结果。

**思路**:
1. 保存forward时的running statistics (L, M)
2. Backward时也分块计算
3. 利用链式法则计算dQ, dK, dV

```python
def flash_attention_backward(dO, Q, K, V, O, L, M, causal=False):
    """
    dO: [batch, n_heads, seq_len, d_head]
    L, M: forward时保存的statistics
    """
    # 类似forward的分块策略
    # 但需要反向传播
    # ...
    return dQ, dK, dV
```

---

## 🧪 测试与验证

### 1. 数值正确性测试
```python
def test_flash_attention():
    batch, n_heads, seq_len, d_head = 2, 4, 1024, 64

    Q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    K = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    V = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')

    # Standard attention
    out_ref = standard_attention(Q, K, V, causal=True)

    # Flash attention
    out_flash = flash_attention(Q, K, V, causal=True)

    # 检查数值
    assert torch.allclose(out_ref, out_flash, rtol=1e-3, atol=1e-3)
    print("✅ 数值测试通过！")
```

### 2. 内存使用测试
```python
def test_memory():
    seq_len = 8192
    Q = torch.randn(1, 8, seq_len, 64, device='cuda')

    # Standard attention内存
    torch.cuda.reset_peak_memory_stats()
    out = standard_attention(Q, Q, Q)
    std_mem = torch.cuda.max_memory_allocated() / 1e9

    # Flash attention内存
    torch.cuda.reset_peak_memory_stats()
    out = flash_attention(Q, Q, Q)
    flash_mem = torch.cuda.max_memory_allocated() / 1e9

    print(f"Standard: {std_mem:.2f} GB")
    print(f"Flash: {flash_mem:.2f} GB")
    print(f"节省: {std_mem/flash_mem:.2f}x")
```

### 3. 速度测试
```python
def test_speed():
    import time

    seq_len = 4096
    Q = torch.randn(8, 12, seq_len, 64, device='cuda')

    # Warmup
    for _ in range(10):
        _ = flash_attention(Q, Q, Q)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = flash_attention(Q, Q, Q)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"平均时间: {elapsed/100*1000:.2f} ms")
```

---

## 📊 性能优化技巧

### 1. 选择合适的block size
```python
# 经验值
BLOCK_M = 64   # 或 128
BLOCK_N = 64   # 或 128

# 需要满足: BLOCK_M * d_head 和 BLOCK_N * d_head 能放进SRAM
# A100: SRAM per SM = ~192KB
```

### 2. 使用Tensor Cores
```python
# Triton自动使用Tensor Cores进行矩阵乘法
# 确保使用float16或bfloat16
Q = Q.to(torch.float16)
```

### 3. 优化内存访问模式
```python
# 确保coalesced memory access
# 使用tl.make_block_ptr等高级特性
```

---

## ⚠️ 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 数值不对 | Online softmax公式错误 | 仔细检查公式推导 |
| OOM | Block size太大 | 减小BLOCK_M/BLOCK_N |
| 速度没有提升 | HBM访问没有减少 | 检查tiling逻辑 |
| Causal mask无效 | Mask逻辑错误 | 检查mask条件 |
| Triton编译错误 | 语法问题 | 参考Triton文档 |

---

## 📚 参考资料

1. **Flash Attention论文**（必读）
   - FlashAttention: Fast and Memory-Efficient Exact Attention

2. **Triton文档**
   - https://triton-lang.org/

3. **Online Softmax**
   - https://arxiv.org/abs/1805.02867

4. **参考实现**
   - Tri Dao's official implementation
   - PyTorch's SDPA (scaled_dot_product_attention)

---

## ✅ 提交检查清单

- [ ] Forward实现正确，数值与标准attention一致
- [ ] 内存使用显著降低（2-4x）
- [ ] 速度有提升（或至少不下降）
- [ ] 支持causal mask
- [ ] 支持不同batch size和head数量
- [ ] 代码有清晰注释
- [ ] 包含测试代码

---

## 💡 进阶方向

1. **Flash Attention V2** - 更好的并行性
2. **Flash Decoding** - 优化推理阶段的KV cache
3. **支持ALiBi/Rotary** - 位置编码
4. **支持GQA/MQA** - Group Query Attention

---

**Flash Attention是CS336最难的作业，但也是最有价值的！加油！** 🚀

