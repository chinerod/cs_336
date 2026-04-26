# CS336 Required Papers Collection

## Foundational Papers (Must Read)

### 1. Transformer Architecture
**Title**: Attention Is All You Need
**Authors**: Vaswani et al. (Google Brain)
**Year**: 2017
**Venue**: NeurIPS
**Link**: https://arxiv.org/abs/1706.03762
**Local File**: 05-Required-Readings/Foundational/Attention-is-All-You-Need.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Original Transformer paper. Foundation of modern NLP.
**Key Concepts**:
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Parallel processing (vs RNNs)

### 2. BERT
**Title**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
**Authors**: Devlin et al. (Google)
**Year**: 2019
**Venue**: NAACL
**Link**: https://arxiv.org/abs/1810.04805
**Local File**: 05-Required-Readings/Foundational/BERT-Pretraining.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Introduced bidirectional pretraining.

### 3. GPT-3
**Title**: Language Models are Few-Shot Learners
**Authors**: Brown et al. (OpenAI)
**Year**: 2020
**Venue**: NeurIPS
**Link**: https://arxiv.org/abs/2005.14165
**Local File**: 05-Required-Readings/Foundational/GPT-3.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Demonstrated emergent abilities at scale.

### 4. GPT-4 Technical Report
**Title**: GPT-4 Technical Report
**Authors**: OpenAI
**Year**: 2023
**Link**: https://arxiv.org/abs/2303.08774
**Local File**: 05-Required-Readings/Foundational/GPT-4-Report.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Latest GPT architecture details.

## Scaling Laws Papers

### 5. OpenAI Scaling Laws
**Title**: Scaling Laws for Neural Language Models
**Authors**: Kaplan et al. (OpenAI)
**Year**: 2020
**Link**: https://arxiv.org/abs/2001.08361
**Local File**: 05-Required-Readings/Scaling-Laws/Scaling-Laws-for-Neural-LMs.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Assignment 3 core reference.
**Key Findings**:
- Loss ∝ (Model Size)^(-0.074)
- Loss ∝ (Data Size)^(-0.095)
- Loss ∝ (Compute)^(-0.057)

### 6. Chinchilla
**Title**: Training Compute-Optimal Large Language Models
**Authors**: Hoffmann et al. (DeepMind)
**Year**: 2022
**Venue**: ICML
**Link**: https://arxiv.org/abs/2203.15556
**Local File**: 05-Required-Readings/Scaling-Laws/Chinchilla-Training-Compute.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Chinchilla scaling laws. Refutes OpenAI scaling.
**Key Findings**:
- Model size and data should scale equally
- GPT-3 was undertrained
- Compute-optimal: 20 tokens per parameter

### 7. Emergent Abilities
**Title**: Emergent Abilities of Large Language Models
**Authors**: Wei et al. (Google)
**Year**: 2022
**Link**: https://arxiv.org/abs/2206.07682
**Local File**: 05-Required-Readings/Scaling-Laws/Emergent-Abilities.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Discussion of emergent capabilities.

## Optimization Papers

### 8. Flash Attention 1
**Title**: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
**Authors**: Dao et al. (Stanford)
**Year**: 2022
**Venue**: NeurIPS
**Link**: https://arxiv.org/abs/2205.14135
**Local File**: 05-Required-Readings/Optimization/FlashAttention-1.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Assignment 2 core reference. Must understand tiling.
**Key Concepts**:
- Tiling algorithm
- Memory-efficient attention
- IO-aware computation

### 9. Flash Attention 2
**Title**: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
**Authors**: Dao (Stanford)
**Year**: 2023
**Link**: https://arxiv.org/abs/2307.08691
**Local File**: 05-Required-Readings/Optimization/FlashAttention-2.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Improved version with better parallelism.

### 10. ZeRO
**Title**: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
**Authors**: Rajbhandari et al. (Microsoft)
**Year**: 2020
**Venue**: SC
**Link**: https://arxiv.org/abs/1910.02054
**Local File**: 05-Required-Readings/Optimization/ZeRO-Infinity.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Distributed training memory optimization.
**Key Concepts**:
- ZeRO stages (1, 2, 3)
- Parameter partitioning
- Gradient partitioning
- Optimizer state partitioning

### 11. Megatron-LM
**Title**: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
**Authors**: Shoeybi et al. (NVIDIA)
**Year**: 2020
**Link**: https://arxiv.org/abs/1909.08053
**Local File**: 05-Required-Readings/Optimization/Megatron-LM.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Model parallelism techniques.

## RLHF & Alignment Papers

### 12. InstructGPT
**Title**: Training Language Models to Follow Instructions with Human Feedback
**Authors**: Ouyang et al. (OpenAI)
**Year**: 2022
**Venue**: NeurIPS
**Link**: https://arxiv.org/abs/2203.02155
**Local File**: 05-Required-Readings/RLHF-Alignment/InstructGPT.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Assignment 5 core reference. Original RLHF paper.
**Key Concepts**:
- SFT (Supervised Fine-Tuning)
- Reward model training
- PPO for language models
- KL divergence penalty

### 13. Constitutional AI
**Title**: Constitutional AI: Harmlessness from AI Feedback
**Authors**: Bai et al. (Anthropic)
**Year**: 2022
**Link**: https://arxiv.org/abs/2212.08073
**Local File**: 05-Required-Readings/RLHF-Alignment/Constitutional-AI.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Alternative to human feedback.

### 14. DPO
**Title**: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
**Authors**: Rafailov et al. (Stanford)
**Year**: 2023
**Link**: https://arxiv.org/abs/2305.18290
**Local File**: 05-Required-Readings/RLHF-Alignment/DPO-Direct-Preference.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Simpler alternative to PPO. CS336 Assignment 5 Part 2.
**Key Concepts**:
- Closed-form policy extraction
- No reward model needed
- Simpler than RLHF

### 15. PPO
**Title**: Proximal Policy Optimization Algorithms
**Authors**: Schulman et al. (OpenAI)
**Year**: 2017
**Link**: https://arxiv.org/abs/1707.06347
**Local File**: 05-Required-Readings/RLHF-Alignment/PPO-Proximal-Policy.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Underlying RL algorithm for RLHF.

## Data Processing Papers

### 16. Deduplication
**Title**: Deduplicating Training Data Makes Language Models Better
**Authors**: Lee et al. (Google)
**Year**: 2022
**Link**: https://arxiv.org/abs/2107.06499
**Local File**: 05-Required-Readings/Data-Processing/Deduplicating-Training-Data.pdf
**Importance**: ⭐⭐⭐⭐⭐
**Notes**: Assignment 4 core reference.
**Key Concepts**:
- MinHash for deduplication
- Near-duplicate detection
- Quality improvement

### 17. The Pile
**Title**: The Pile: An 800GB Dataset of Diverse Text for Language Modeling
**Authors**: Gao et al.
**Year**: 2020
**Link**: https://arxiv.org/abs/2101.00027
**Local File**: 05-Required-Readings/Data-Processing/The-Pile.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Large-scale dataset construction.

### 18. Data Mixture Laws
**Title**: Scaling Laws for Data Mixing
**Authors**: Albalak et al.
**Year**: 2023
**Link**: https://arxiv.org/abs/2311.09601
**Local File**: 05-Required-Readings/Data-Processing/Data-Mixture-Laws.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: How to mix different data sources.

### 19. Quality Filtering
**Title**: Quality Data is All You Need
**Authors**: Li et al.
**Year**: 2024
**Link**: https://arxiv.org/abs/2405.09818
**Local File**: 05-Required-Readings/Data-Processing/Quality-Filtering.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Data quality vs quantity.

## Additional Important Papers

### 20. Tokenization
**Title**: Neural Machine Translation of Rare Words with Subword Units
**Authors**: Sennrich et al.
**Year**: 2016
**Link**: https://arxiv.org/abs/1508.07909
**Local File**: 05-Required-Readings/Foundational/BPE-Tokenization.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: BPE algorithm original paper.

### 21. LLaMA
**Title**: LLaMA: Open and Efficient Foundation Language Models
**Authors**: Touvron et al. (Meta)
**Year**: 2023
**Link**: https://arxiv.org/abs/2302.13971
**Local File**: 06-Supplementary-Readings/Architecture/LLaMA.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Popular open-source architecture.

### 22. PaLM
**Title**: PaLM: Scaling Language Modeling with Pathways
**Authors**: Chowdhery et al. (Google)
**Year**: 2022
**Link**: https://arxiv.org/abs/2204.02311
**Local File**: 06-Supplementary-Readings/Architecture/PaLM.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: 540B parameter model.

### 23. Mistral
**Title**: Mistral 7B
**Authors**: Jiang et al.
**Year**: 2023
**Link**: https://arxiv.org/abs/2310.06825
**Local File**: 06-Supplementary-Readings/Architecture/Mistral.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Efficient 7B model with sliding window attention.

### 24. MoE
**Title**: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
**Authors**: Fedus et al. (Google)
**Year**: 2022
**Link**: https://arxiv.org/abs/2101.03961
**Local File**: 06-Supplementary-Readings/Architecture/Mixtral-MoE.pdf
**Importance**: ⭐⭐⭐⭐
**Notes**: Mixture of Experts architecture.

## Paper Reading Strategy

### Priority 1 (Read First)
1. Attention Is All You Need
2. BERT
3. GPT-3
4. Flash Attention 1 & 2
5. InstructGPT
6. Scaling Laws (both OpenAI and Chinchilla)

### Priority 2 (Read During Course)
7. ZeRO
8. Megatron-LM
9. DPO
10. Deduplication paper
11. Tokenization papers

### Priority 3 (Reference As Needed)
12. Architecture papers (LLaMA, PaLM, Mistral)
13. Additional optimization papers
14. Survey papers

## Download Script

Create `download-papers.sh`:

```bash
#!/bin/bash
# Download all CS336 required papers

BASE_DIR="05-Required-Readings"

# Create directories
mkdir -p $BASE_DIR/{Foundational,Scaling-Laws,Optimization,RLHF-Alignment,Data-Processing}

# Foundational Papers
cd $BASE_DIR/Foundational
wget -O Attention-is-All-You-Need.pdf https://arxiv.org/pdf/1706.03762.pdf
wget -O BERT-Pretraining.pdf https://arxiv.org/pdf/1810.04805.pdf
wget -O GPT-3.pdf https://arxiv.org/pdf/2005.14165.pdf
wget -O GPT-4-Report.pdf https://arxiv.org/pdf/2303.08774.pdf
wget -O BPE-Tokenization.pdf https://arxiv.org/pdf/1508.07909.pdf

cd ../..

# Scaling Laws
cd $BASE_DIR/Scaling-Laws
wget -O Scaling-Laws-for-Neural-LMs.pdf https://arxiv.org/pdf/2001.08361.pdf
wget -O Chinchilla-Training-Compute.pdf https://arxiv.org/pdf/2203.15556.pdf
wget -O Emergent-Abilities.pdf https://arxiv.org/pdf/2206.07682.pdf

cd ../..

# Optimization
cd $BASE_DIR/Optimization
wget -O FlashAttention-1.pdf https://arxiv.org/pdf/2205.14135.pdf
wget -O FlashAttention-2.pdf https://arxiv.org/pdf/2307.08691.pdf
wget -O ZeRO-Infinity.pdf https://arxiv.org/pdf/1910.02054.pdf
wget -O Megatron-LM.pdf https://arxiv.org/pdf/1909.08053.pdf

cd ../..

# RLHF Alignment
cd $BASE_DIR/RLHF-Alignment
wget -O InstructGPT.pdf https://arxiv.org/pdf/2203.02155.pdf
wget -O Constitutional-AI.pdf https://arxiv.org/pdf/2212.08073.pdf
wget -O DPO-Direct-Preference.pdf https://arxiv.org/pdf/2305.18290.pdf
wget -O PPO-Proximal-Policy.pdf https://arxiv.org/pdf/1707.06347.pdf

cd ../..

# Data Processing
cd $BASE_DIR/Data-Processing
wget -O Deduplicating-Training-Data.pdf https://arxiv.org/pdf/2107.06499.pdf
wget -O The-Pile.pdf https://arxiv.org/pdf/2101.00027.pdf
wget -O Data-Mixture-Laws.pdf https://arxiv.org/pdf/2311.09601.pdf
wget -O Quality-Filtering.pdf https://arxiv.org/pdf/2405.09818.pdf

cd ../..

echo "All papers downloaded!"
```

## Paper Notes Template

For each paper, create notes:

```markdown
# Paper Title

## Metadata
- Authors:
- Year:
- Venue:
- Link:

## Key Contributions
1.
2.
3.

## Main Results
-

## Technical Details
### Method
### Architecture
### Training

## My Notes
- Important equations:
- Questions:
- Connections to CS336:

## Relevance to Assignments
- Assignment X:

## Code Implementation Notes
- Can implement:
- Difficult parts:
```

---

**Total Papers**: 24 core papers
**Estimated Reading Time**: 60-80 hours
**Priority Papers**: 12 (complete these first)
