# Tokenization 研究资料

## 🎯 核心概念

Tokenization是将文本映射到离散token序列的过程。

---

## 📄 必读论文

### 经典
1. **"Neural Machine Translation of Rare Words with Subword Units"** - Sennrich et al., 2016
   - BPE算法
   - https://arxiv.org/abs/1508.07909

2. **"SentencePiece: A simple and language independent subword tokenizer"** - Kudo & Richardson, 2018
   - https://arxiv.org/abs/1808.06226

3. **"Google's Neural Machine Translation System"** - Wu et al., 2016
   - WordPiece
   - https://arxiv.org/abs/1609.08144

### 近期研究
4. **"BPE Dropout: Simple and Effective Subword Regularization"** - Provilkov et al., 2020
   - https://arxiv.org/abs/1910.13267

5. **"From Characters to Words: A Brief History of Tokenization in NLP"** - Mielke et al., 2021
   - 综述
   - https://arxiv.org/abs/2112.10508

---

## 📚 推荐书籍章节

- **"Speech and Language Processing"** - Jurafsky & Martin
  - Chapter 2: Regular Expressions, Text Normalization, Edit Distance
  - 免费: https://web.stanford.edu/~jurafsky/slp3/

---

## 🔗 代码实现

### 官方
- **SentencePiece**: https://github.com/google/sentencepiece
- **Hugging Face Tokenizers**: https://github.com/huggingface/tokenizers

### 参考
- **GPT-2 BPE**: https://github.com/openai/gpt-2/blob/master/src/encoder.py

---

## 💡 研究方向

- 多语言Tokenization公平性
- 自适应Vocabulary大小
- 视觉Tokenization (VQ-VAE)

---

*Week 1 研究资料*
