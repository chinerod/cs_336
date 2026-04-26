# WordPiece Tokenizer

```python
from collections import defaultdict
import re

class WordPieceTokenizer:
    """
    WordPiece Tokenizer (BERT使用)
    """

    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
        self.max_input_chars_per_word = 100

    def train(self, corpus):
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

        print(f"Initial vocab size: {len(self.vocab)}")

        # 迭代添加新token
        while len(self.vocab) < self.vocab_size:
            pairs = defaultdict(int)

            # 统计所有可能的pair
            for word, freq in word_freqs.items():
                symbols = list(word)
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i+1])] += freq

            if not pairs:
                break

            # 选择频率最高的pair
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

                # 更新word_freqs
                new_word_freqs = {}
                for word, freq in word_freqs.items():
                    new_word = word.replace(new_token, new_token)
                    new_word_freqs[new_word] = freq
                word_freqs = new_word_freqs

                if len(self.vocab) % 1000 == 0:
                    print(f"Vocab size: {len(self.vocab)}")

    def encode(self, text):
        """编码文本"""
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

    def decode(self, token_ids):
        """解码"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '[UNK]') for id in token_ids]

        text = ' '.join(tokens)
        text = text.replace(' ##', '')
        text = text.replace('##', '')
        return text


# 使用示例
if __name__ == '__main__':
    corpus = [
        "hello world",
        "hello there",
        "world peace",
        "hello world peace"
    ]

    tokenizer = WordPieceTokenizer(vocab_size=50)
    tokenizer.train(corpus)

    text = "hello world"
    ids = tokenizer.encode(text)
    print(f"Encoded: {ids}")

    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
```

---

*WordPiece Tokenizer实现*
