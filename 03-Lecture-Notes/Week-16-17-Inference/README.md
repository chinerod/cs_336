# 推理优化研究资料

## 🎯 核心概念

KV Cache、量化、投机解码、服务系统

---

## 📄 必读论文

### KV Cache优化
1. **"vLLM: Efficient Memory Management for LLM Serving"** - Kwon et al., 2023
   - https://arxiv.org/abs/2309.06180

2. **"Efficient Large Language Models: A Survey"** - Zhu et al., 2023
   - https://arxiv.org/abs/2312.03863

### 量化
3. **"LLM.int8(): 8-bit Matrix Multiplication"** - Dettmers et al., 2022
   - https://arxiv.org/abs/2208.07339

4. **"AWQ: Activation-aware Weight Quantization"** - Lin et al., 2023
   - https://arxiv.org/abs/2306.00978

### 投机解码
5. **"Fast Inference from Transformers via Speculative Decoding"** - Leviathan et al., 2022
   - https://arxiv.org/abs/2211.17192

6. **"Medusa: Simple LLM Inference Acceleration Framework"** - Cai et al., 2024
   - https://arxiv.org/abs/2401.10774

---

## 📄 服务系统

7. **"Orca: A Distributed Serving System"** - Yu et al., 2022
   - https://www.usenix.org/conference/osdi22/presentation/yu

8. **"FasterTransformer"** - NVIDIA
   - https://github.com/NVIDIA/FasterTransformer

---

## 🔗 代码

- **vLLM**: https://github.com/vllm-project/vllm
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
- **DeepSpeed Inference**: https://github.com/microsoft/DeepSpeed

---

## 💡 研究方向

- 动态batching
- 前缀缓存
- 边缘设备部署

---

*Week 16-17 研究资料*
