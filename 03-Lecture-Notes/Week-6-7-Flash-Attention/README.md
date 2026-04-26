# Flash Attention 研究资料

## 🎯 核心概念

通过Tiling和Online Softmax减少HBM访问，实现内存高效的Attention

---

## 📄 必读论文

### Flash Attention
1. **"FlashAttention: Fast and Memory-Efficient Exact Attention"** - Dao et al., 2022
   - https://arxiv.org/abs/2205.14135

2. **"FlashAttention-2: Faster Attention with Better Parallelism"** - Dao, 2023
   - https://arxiv.org/abs/2307.08691

### Online Softmax
3. **"Online normalizer calculation for softmax"** - Milakov & Gimelshein, 2018
   - https://arxiv.org/abs/1805.02867

### PageAttention
4. **"Efficient Memory Management for Large Language Model Serving"** - Kwon et al., 2023 (vLLM)
   - https://arxiv.org/abs/2309.06180

---

## 📄 相关工作

5. **"Memory-Efficient Attention"** - Rabe & Staats, 2021
   - https://arxiv.org/abs/2112.05682

6. **"The I/O Complexity of Attention"** - Dao et al., 2023
   - 理论分析

---

## 🔗 代码实现

### 官方
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **Triton Tutorial**: https://triton-lang.org/main/getting-started/tutorials/index.html

### 参考
- **xFormers**: https://github.com/facebookresearch/xformers

---

## 🎓 学习资源

### Triton
- **Triton Documentation**: https://triton-lang.org/
- **OpenAI Triton Tutorial**: https://github.com/openai/triton/tree/main/python/tutorials

---

## 💡 研究方向

- 稀疏Flash Attention
- 多模态Flash Attention
- Kernel自动调优

---

*Week 6-7 研究资料*
