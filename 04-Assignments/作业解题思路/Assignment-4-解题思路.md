# Assignment 4 解题思路指南

## 📝 作业概述

**主题**: 数据处理与清洗
**目标**: 构建完整的数据处理pipeline，包括去重、过滤、质量评估
**难度**: ⭐⭐⭐⭐（工程量大）
**预计时间**: 2-3周
**核心技能**: 大数据处理、去重算法、并行计算

---

## 📚 知识准备

### 1. Common Crawl数据格式

**WARC格式**:
```
WARC/1.0
WARC-Type: response
WARC-Target-URI: http://example.com/
Content-Length: 1234

HTTP/1.1 200 OK
Content-Type: text/html

<html>...
```

**WET格式** (提取后的纯文本):
```
WARC/1.0
WARC-Type: conversion
WARC-Target-URI: http://example.com/
Content-Length: 567

纯文本内容...
```

### 2. MinHash算法

**原理**: 用于快速估计集合的Jaccard相似度

```python
# Jaccard相似度
J(A, B) = |A ∩ B| / |A ∪ B|

# MinHash估计
- 使用k个hash函数
- 对每个集合，计算k个最小hash值（签名）
- 比较签名相同的比例 ≈ Jaccard相似度
```

### 3. LSH (Locality Sensitive Hashing)

**作用**: 快速找到相似文档对（无需两两比较）

**原理**: 将相似的文档映射到同一个bucket

---

## 🗂️ 作业结构

### Part 1: 数据提取

#### 1.1 下载Common Crawl数据

```python
import requests
from warcio import ArchiveIterator

def download_cc_segment(cc_url, output_dir):
    """下载Common Crawl segment"""
    # CC路径格式
    # https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-50/...

    response = requests.get(cc_url, stream=True)

    with open(f"{output_dir}/segment.warc.gz", 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_text_from_warc(warc_path):
    """从WARC提取文本"""
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.rec_type == 'response':
                uri = record.rec_headers.get_header('WARC-Target-URI')
                content = record.content_stream().read()

                # 提取文本（可用trafilatura或beautifulsoup）
                text = extract_text(content)

                yield {'uri': uri, 'text': text}
```

#### 1.2 文本清洗

```python
import re
from urllib.parse import urlparse

def clean_text(text):
    """清洗文本"""
    # 1. 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 2. 规范化空白
    text = re.sub(r'\s+', ' ', text)

    # 3. 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)

    # 4. 规范化Unicode
    text = text.strip()

    return text

def is_valid_document(doc):
    """检查文档是否有效"""
    # 长度检查
    if len(doc['text']) < 100:
        return False

    # 语言检查（可选）
    if not is_english(doc['text']):
        return False

    # 垃圾内容过滤
    if is_garbage(doc['text']):
        return False

    return True
```

---

### Part 2: MinHash去重

#### 2.1 Shingling（生成n-gram集合）

```python
def get_shingles(text, k=5):
    """
    将文本转换为k-shingle集合

    k=5表示5个连续的token/word
    """
    words = text.split()
    shingles = set()

    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(shingle)

    return shingles

# 示例
# text = "the cat sat on the mat"
# k=3: {"the cat sat", "cat sat on", "sat on the", "on the mat"}
```

#### 2.2 MinHash签名计算

```python
import hashlib
import random

class MinHash:
    def __init__(self, num_hashes=128):
        self.num_hashes = num_hashes

        # 生成随机hash函数参数
        # h(x) = (a*x + b) % p
        self.a = [random.randint(1, 2**32) for _ in range(num_hashes)]
        self.b = [random.randint(0, 2**32) for _ in range(num_hashes)]
        self.p = 2**32 - 1  # 大素数

    def _hash(self, shingle, i):
        """第i个hash函数"""
        x = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        return (self.a[i] * x + self.b[i]) % self.p

    def compute_signature(self, shingles):
        """计算MinHash签名"""
        signature = []

        for i in range(self.num_hashes):
            # 对每个hash函数，找最小的hash值
            min_hash = min(self._hash(s, i) for s in shingles)
            signature.append(min_hash)

        return signature

    def jaccard_estimate(self, sig1, sig2):
        """估计Jaccard相似度"""
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.num_hashes
```

#### 2.3 LSH实现

```python
class LSH:
    def __init__(self, num_bands=16, rows_per_band=8):
        """
        将签名分成bands
        num_bands * rows_per_band = num_hashes
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [{} for _ in range(num_bands)]

    def _hash_band(self, band):
        """hash一个band"""
        return hash(tuple(band))

    def add_signature(self, doc_id, signature):
        """添加文档签名"""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]

            band_hash = self._hash_band(band)

            if band_hash not in self.buckets[band_idx]:
                self.buckets[band_idx][band_hash] = []
            self.buckets[band_idx][band_hash].append(doc_id)

    def get_candidates(self, signature):
        """获取可能的相似文档"""
        candidates = set()

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]

            band_hash = self._hash_band(band)

            if band_hash in self.buckets[band_idx]:
                candidates.update(self.buckets[band_idx][band_hash])

        return candidates
```

#### 2.4 完整去重流程

```python
def deduplicate_documents(documents, threshold=0.8):
    """
    文档去重主函数

    Args:
        documents: 文档列表，每个文档是dict，包含id和text
        threshold: Jaccard相似度阈值
    """
    minhash = MinHash(num_hashes=128)
    lsh = LSH(num_bands=16, rows_per_band=8)

    # 第一步：计算所有文档的签名并加入LSH
    signatures = {}
    for doc in documents:
        shingles = get_shingles(doc['text'], k=5)
        sig = minhash.compute_signature(shingles)
        signatures[doc['id']] = sig
        lsh.add_signature(doc['id'], sig)

    # 第二步：找到相似文档对
    duplicates = set()

    for doc in documents:
        if doc['id'] in duplicates:
            continue

        sig = signatures[doc['id']]
        candidates = lsh.get_candidates(sig)

        for candidate_id in candidates:
            if candidate_id == doc['id'] or candidate_id in duplicates:
                continue

            # 精确计算相似度
            sim = minhash.jaccard_estimate(sig, signatures[candidate_id])

            if sim > threshold:
                duplicates.add(candidate_id)

    # 返回非重复文档
    return [doc for doc in documents if doc['id'] not in duplicates]
```

---

### Part 3: 质量过滤

#### 3.1 质量评分

```python
import re

def quality_score(text):
    """
    计算文档质量分数
    返回0-1之间的值，越高越好
    """
    scores = []

    # 1. 长度分数
    word_count = len(text.split())
    if word_count < 50:
        scores.append(0)
    elif word_count > 10000:
        scores.append(0.5)  # 太长可能有问题
    else:
        scores.append(1.0)

    # 2. 平均词长（过滤乱码）
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    if 3 < avg_word_len < 10:
        scores.append(1.0)
    else:
        scores.append(0.3)

    # 3. 符号比例（过滤代码、垃圾）
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    scores.append(alpha_ratio)

    # 4. 行尾标点比例（过滤列表、表格）
    lines = text.split('\n')
    lines_with_ending_punct = sum(1 for l in lines if re.search(r'[.!?]$', l.strip()))
    punct_ratio = lines_with_ending_punct / len(lines) if lines else 0
    scores.append(min(punct_ratio * 2, 1.0))  # 调整权重

    # 加权平均
    weights = [0.3, 0.2, 0.3, 0.2]
    final_score = sum(s * w for s, w in zip(scores, weights))

    return final_score
```

#### 3.2 语言检测

```python
# 使用langdetect或fasttext
from langdetect import detect

def detect_language(text):
    """检测文本语言"""
    try:
        lang = detect(text[:1000])  # 只检测前1000字符
        return lang
    except:
        return 'unknown'

def filter_by_language(documents, target_lang='en'):
    """按语言过滤"""
    return [doc for doc in documents if detect_language(doc['text']) == target_lang]
```

#### 3.3 敏感内容过滤

```python
def contains_sensitive_content(text):
    """检查是否包含敏感/不当内容"""
    # 可以加载敏感词列表
    # 或使用预训练分类器

    # 示例：简单检查
    sensitive_patterns = [
        r'\b(drug|porn|hate)\b',
        # 更多模式...
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False
```

---

### Part 4: 完整Pipeline

#### 4.1 并行处理

```python
from multiprocessing import Pool
from functools import partial

def process_batch(batch, minhash, lsh):
    """处理一个batch的文档"""
    results = []

    for doc in batch:
        # 清洗
        doc['text'] = clean_text(doc['text'])

        # 质量检查
        if not is_valid_document(doc):
            continue

        # 质量评分
        doc['quality_score'] = quality_score(doc['text'])
        if doc['quality_score'] < 0.5:
            continue

        # 计算MinHash签名
        shingles = get_shingles(doc['text'])
        sig = minhash.compute_signature(shingles)
        doc['signature'] = sig

        results.append(doc)

    return results

def parallel_process(documents, num_workers=8):
    """并行处理文档"""
    # 分成batches
    batch_size = 1000
    batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

    minhash = MinHash()
    lsh = LSH()

    with Pool(num_workers) as pool:
        func = partial(process_batch, minhash=minhash, lsh=lsh)
        results = pool.map(func, batches)

    # 合并结果
    all_docs = []
    for batch_result in results:
        all_docs.extend(batch_result)

    return all_docs
```

#### 4.2 完整Pipeline

```python
class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.minhash = MinHash(num_hashes=config['num_hashes'])
        self.lsh = LSH(num_bands=config['num_bands'])

    def run(self, input_path, output_path):
        """运行完整pipeline"""
        # 1. 加载数据
        print("Loading data...")
        documents = self._load_data(input_path)
        print(f"Loaded {len(documents)} documents")

        # 2. 清洗和过滤
        print("Cleaning and filtering...")
        documents = self._clean_and_filter(documents)
        print(f"After filtering: {len(documents)} documents")

        # 3. 去重
        print("Deduplicating...")
        documents = self._deduplicate(documents)
        print(f"After dedup: {len(documents)} documents")

        # 4. 质量排序
        print("Sorting by quality...")
        documents = sorted(documents, key=lambda x: x['quality_score'], reverse=True)

        # 5. 保存
        print("Saving...")
        self._save(documents, output_path)

        return documents

    def _load_data(self, path):
        """加载数据"""
        # 根据格式选择加载方式
        pass

    def _clean_and_filter(self, documents):
        """清洗和过滤"""
        valid_docs = []
        for doc in documents:
            # 清洗
            doc['text'] = clean_text(doc['text'])

            # 基本检查
            if not is_valid_document(doc):
                continue

            # 质量评分
            doc['quality_score'] = quality_score(doc['text'])
            if doc['quality_score'] < self.config['quality_threshold']:
                continue

            # 语言过滤
            if self.config.get('language'):
                if detect_language(doc['text']) != self.config['language']:
                    continue

            valid_docs.append(doc)

        return valid_docs

    def _deduplicate(self, documents):
        """去重"""
        # 计算签名
        for doc in documents:
            shingles = get_shingles(doc['text'], self.config.get('shingle_size', 5))
            doc['signature'] = self.minhash.compute_signature(shingles)
            self.lsh.add_signature(doc['id'], doc['signature'])

        # 去重
        duplicates = set()
        for doc in documents:
            if doc['id'] in duplicates:
                continue

            candidates = self.lsh.get_candidates(doc['signature'])
            for candidate_id in candidates:
                if candidate_id <= doc['id']:  # 避免重复检查
                    continue

                sim = self.minhash.jaccard_estimate(
                    doc['signature'],
                    next(d['signature'] for d in documents if d['id'] == candidate_id)
                )

                if sim > self.config['dedup_threshold']:
                    duplicates.add(candidate_id)

        return [d for d in documents if d['id'] not in duplicates]

    def _save(self, documents, path):
        """保存结果"""
        import json
        with open(path, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
```

---

## 📊 评估指标

### 去重效果
```python
def evaluate_deduplication(original_count, deduped_count, sample_size=1000):
    """评估去重效果"""
    # 随机抽样检查
    # 人工或自动判断是否正确去重

    precision = ...  # 正确去重 / 总去重
    recall = ...     # 正确去重 / 应该去重

    return {
        'dedup_ratio': 1 - deduped_count / original_count,
        'precision': precision,
        'recall': recall,
    }
```

### 质量分布
```python
def plot_quality_distribution(documents):
    """绘制质量分数分布"""
    scores = [d['quality_score'] for d in documents]
    plt.hist(scores, bins=50)
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.title('Quality Score Distribution')
    plt.savefig('quality_dist.png')
```

---

## ⚠️ 常见陷阱

| 问题 | 原因 | 解决 |
|------|------|------|
| 内存不足 | 加载太多文档 | 使用streaming处理 |
| 去重不彻底 | 阈值太高 | 调整threshold |
| 误删文档 | shingle size不合适 | 调整k值 |
| 处理太慢 | 没有并行 | 使用multiprocessing |

---

## ✅ 提交检查清单

- [ ] 数据提取正确
- [ ] MinHash签名计算正确
- [ ] LSH去重有效
- [ ] 质量评分合理
- [ ] 并行处理实现
- [ ] 完整pipeline可运行
- [ ] 处理10GB+数据

---

## 💡 进阶方向

1. **SimHash** - 另一种去重算法
2. **语义去重** - 用embedding判断相似
3. **增量处理** - 支持streaming数据
4. **分布式处理** - 用Spark处理TB级数据

---

**数据处理是LLM训练的基础，认真做！** 📊

