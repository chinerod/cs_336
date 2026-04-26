# Tokenization 实现

## 📚 BPE Tokenizer

```python
"""
Byte Pair Encoding 实现

参考: Neural Machine Translation of Rare Words with Subword Units
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple


class BPETokenizer:
    """BPE Tokenizer实现"""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, corpus: List[str]) -> None:
        """
        训练BPE

        Args:
            corpus: 训练语料，字符串列表
        """
        # 1. 预处理：空格分隔
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.strip().split()
            for word in words:
                # 字符分割，末尾加</w>
                word_freqs[' '.join(word) + ' </w>'] += 1

        # 2. 初始化词汇表
        char_vocab = set()
        for word in word_freqs:
            for char in word.split():
                char_vocab.add(char)
        self.vocab = {char: i for i, char in enumerate(sorted(char_vocab))}

        # 3. BPE合并
        num_merges = self.vocab_size - len(self.vocab)

        for i in range(num_merges):
            # 统计pair频率
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j+1])] += freq

            if not pairs:
                break

            # 找最频繁的pair
            best = max(pairs, key=pairs.get)

            # 合并
            self.merges.append(best)
            new_vocab = ''.join(best)
            if new_vocab not in self.vocab:
                self.vocab[new_vocab] = len(self.vocab)

            # 更新word_freqs
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = self._merge_word(word, best)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

            print(f"Merge {i+1}/{num_merges}: {best} -> {new_vocab}")

    def _merge_word(self, word: str, pair: Tuple[str, str]) -> str:
        """合并word中的一个pair"""
        symbols = word.split()
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                new_symbols.append(''.join(pair))
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return ' '.join(new_symbols)

    def encode(self, text: str) -> List[int]:
        """
        编码文本

        Args:
            text: 输入文本

        Returns:
            token ids列表
        """
        words = text.strip().split()
        token_ids = []

        for word in words:
            word = ' '.join(word) + ' </w>'

            # 应用所有merge
            for merge in self.merges:
                word = self._merge_word(word, merge)

            # 转换为ids
            tokens = word.split()
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # 未知token，用字符级别
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        解码token ids

        Args:
            token_ids: token id列表

        Returns:
            解码文本
        """
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<unk>') for id in token_ids]

        # 合并
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')

        return text.strip()

    def save(self, path: str) -> None:
        """保存tokenizer"""
        import json
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """加载tokenizer"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = [tuple(m) for m in data['merges']]


# 使用示例
if __name__ == '__main__':
    # 训练语料
    corpus = [
        "low lower lowest",
        "high higher highest",
        "fast faster fastest"
    ]

    # 训练
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)

    # 编码
    text = "low high"
    ids = tokenizer.encode(text)
    print(f"Encoded: {ids}")

    # 解码
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
```

---

## 📚 WordPiece Tokenizer

```python
"""
WordPiece Tokenizer (BERT使用)

与BPE区别：
- 基于语言模型似然选择merge
- 不是最高频，而是最大化训练数据似然
"""

import re
from collections import defaultdict
import math


class WordPieceTokenizer:
    """WordPiece Tokenizer"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
        self.max_input_chars_per_word = 100

    def train(self, corpus: List[str]) -> None:
        """训练WordPiece"""
        # 统计词频
        word_freqs = defaultdict(int)
        for text in corpus:
            for word in text.split():
                word_freqs[word] += 1

        # 初始化字符词汇表
        char_vocab = set()
        for word in word_freqs:
            for char in word:
                char_vocab.add(char)

        for char in sorted(char_vocab):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # 迭代添加新token
        while len(self.vocab) < self.vocab_size:
            best_score = None
            best_pair = None

            # 遍历所有可能的pair
            candidates = self._get_candidates(word_freqs)

            for pair in candidates:
                score = self._score_pair(pair, word_freqs)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            # 添加到词汇表
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            print(f"Added: {new_token}, vocab size: {len(self.vocab)}")

    def _get_candidates(self, word_freqs):
        """获取候选pair"""
        candidates = set()
        for word in word_freqs:
            symbols = list(word)
            for i in range(len(symbols) - 1):
                candidates.add((symbols[i], symbols[i+1]))
        return candidates

    def _score_pair(self, pair, word_freqs):
        """计算pair的分数（似然增益）"""
        # 简化版：使用频率
        freq = 0
        for word, count in word_freqs.items():
            if pair[0] + pair[1] in word:
                freq += count
        return freq

    def encode(self, text: str) -> List[int]:
        """编码"""
        output_tokens = []
        for word in text.split():
            if len(word) > self.max_input_chars_per_word:
                output_tokens.append(self.vocab.get('[UNK]', 1))
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(word):
                end = len(word)
                cur_substr = None

                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = '##' + substr

                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break

                sub_tokens.append(self.vocab[cur_substr])
                start = end

            if is_bad:
                output_tokens.append(self.vocab.get('[UNK]', 1))
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens
```

---

## 📚 SentencePiece

```python
"""
SentencePiece Wrapper

SentencePiece是Google的语言无关tokenizer
支持BPE和Unigram
"""

import sentencepiece as spm


def train_sentencepiece(input_file, model_prefix, vocab_size=32000):
    """
    训练SentencePiece模型

    Args:
        input_file: 输入文本文件
        model_prefix: 模型文件前缀
        vocab_size: 词汇表大小
    """
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # 或 'unigram'
        character_coverage=0.9995,
        num_threads=8,
        split_by_whitespace=True,
        split_by_unicode_script=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name='identity'
    )


class SentencePieceTokenizer:
    """SentencePiece Tokenizer封装"""

    def __init__(self, model_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    def encode(self, text: str) -> List[int]:
        """编码为token ids"""
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids: List[int]) -> str:
        """解码token ids"""
        return self.sp.decode(token_ids)

    def encode_as_pieces(self, text: str) -> List[str]:
        """编码为token字符串"""
        return self.sp.encode(text, out_type=str)

    @property
    def vocab_size(self):
        return len(self.sp)

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def unk_id(self):
        return self.sp.unk_id()
```

---

*Tokenization代码参考*

