# Memory Efficient Attention

```python
import torch
import torch.nn.functional as F

def memory_efficient_attention(Q, K, V, chunk_size=1024):
    """
    内存高效的Attention实现
    通过分块计算减少内存占用
    """
    N, d = Q.shape[-2], Q.shape[-1]
    O = torch.zeros_like(Q)

    for i in range(0, N, chunk_size):
        Qi = Q[..., i:i+chunk_size, :]

        # 逐块计算K, V
        scores = []
        for j in range(0, N, chunk_size):
            Kj = K[..., j:j+chunk_size, :]
            scores.append(Qi @ Kj.transpose(-2, -1) / (d ** 0.5))

        scores = torch.cat(scores, dim=-1)
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        output = []
        for j in range(0, N, chunk_size):
            Vj = V[..., j:j+chunk_size, :]
            aj = attn[..., j:j+chunk_size]
            output.append(aj @ Vj)

        O[..., i:i+chunk_size, :] = sum(output)

    return O
```

---

*内存高效Attention实现*
