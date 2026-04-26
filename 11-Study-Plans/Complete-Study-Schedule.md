# CS336 Study Schedule & Progress Tracker
# Stanford CS336: Language Learning from Scratch

## Overview
- **Course Duration**: 10-12 weeks (flexible)
- **Recommended Study Time**: 12-15 hours/week
- **Total Assignments**: 5
- **Difficulty**: Advanced (5-unit course)

## Path 1: Intensive 10-Week Plan (Full-time)

### Week 1: Introduction & Tokenization
**Topics:**
- Course overview and logistics
- Introduction to Language Models
- Tokenization algorithms (BPE, WordPiece)
- PyTorch refresher

**Tasks:**
- [ ] Watch Week 0 & Week 1 lectures
- [ ] Read: "Neural Machine Translation of Rare Words with Subword Units"
- [ ] Complete tokenization implementation
- [ ] Start Assignment 1

**Deliverables:**
- Basic tokenizer implementation
- Understanding of BPE algorithm

**Time Commitment**: 12-15 hours

---

### Week 2: PyTorch & Resource Management
**Topics:**
- PyTorch advanced features (einops)
- Roofline model
- Memory hierarchy
- Training vs Inference profiling

**Tasks:**
- [ ] Watch Week 2 lectures
- [ ] Learn einops operations
- [ ] Profile simple PyTorch models
- [ ] Continue Assignment 1

**Readings:**
- "Efficient Large-Scale Language Model Training on GPU Clusters"
- Roofline model documentation

**Deliverables:**
- Resource profiling scripts
- Memory usage analysis

---

### Week 3: Architectures
**Topics:**
- Transformer architecture deep dive
- Hyperparameter tuning
- Model configurations

**Tasks:**
- [ ] Watch Week 3 lectures
- [ ] Read: "Attention Is All You Need"
- [ ] Implement basic Transformer
- [ ] Complete Assignment 1 (Due this week)

**Key Papers:**
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers"

**Deliverables:**
- Working Transformer implementation
- Assignment 1 submission

---

### Week 4: Advanced Architectures
**Topics:**
- Mixture of Experts (MoE)
- GPT architecture variants
- Scaling considerations

**Tasks:**
- [ ] Watch Week 4 lectures
- [ ] Read MoE papers
- [ ] Start Assignment 2
- [ ] Begin implementing attention optimizations

**Readings:**
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts"
- "Switch Transformers: Scaling to Trillion Parameter Models"

**Deliverables:**
- MoE understanding document
- Attention optimization plan

---

### Week 5: Hardware & Kernels
**Topics:**
- GPU architecture (CUDA)
- TPU basics
- Kernel optimization
- Triton introduction

**Tasks:**
- [ ] Watch Week 5 lectures
- [ ] Learn Triton basics
- [ ] Continue Assignment 2

**Resources:**
- Triton documentation
- CUDA programming guide

**Deliverables:**
- Triton kernel experiments

---

### Week 6: Systems & Parallelism
**Topics:**
- Distributed training
- Data parallelism
- Model parallelism
- Pipeline parallelism

**Tasks:**
- [ ] Watch Week 6 lectures
- [ ] Read: "Megatron-LM: Training Multi-Billion Parameter Language Models"
- [ ] Complete Assignment 2 (Due this week)
- [ ] Start Assignment 3

**Key Papers:**
- "Megatron-LM"
- "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
- "FlashAttention: Fast and Memory-Efficient Exact Attention"

**Deliverables:**
- Flash Attention implementation
- Distributed training setup
- Assignment 2 submission

---

### Week 7: Scaling Laws
**Topics:**
- Scaling laws theory
- Compute-optimal training
- Chinchilla scaling

**Tasks:**
- [ ] Watch Week 7 lectures
- [ ] Read: "Scaling Laws for Neural Language Models"
- [ ] Continue Assignment 3

**Key Papers:**
- "Scaling Laws for Neural Language Models" (OpenAI)
- "Training Compute-Optimal Large Language Models" (Chinchilla)

**Deliverables:**
- Scaling law experiments
- Model size predictions

---

### Week 8: Evaluation
**Topics:**
- Language model evaluation
- Benchmarks (GLUE, SuperGLUE, etc.)
- Human evaluation

**Tasks:**
- [ ] Watch Week 8 lectures
- [ ] Complete Assignment 3 (Due this week)
- [ ] Start Assignment 4

**Readings:**
- "GLUE: A Multi-Task Benchmark and Analysis Platform"
- "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models"

**Deliverables:**
- Evaluation framework
- Assignment 3 submission

---

### Week 9: Data Engineering
**Topics:**
- Data sources
- Data cleaning
- Filtering strategies
- Deduplication

**Tasks:**
- [ ] Watch Week 9 lectures
- [ ] Read data processing papers
- [ ] Continue Assignment 4

**Key Papers:**
- "Deduplicating Training Data Makes Language Models Better"
- "The Pile: An 800GB Dataset of Diverse Text"

**Deliverables:**
- Data pipeline implementation

---

### Week 10: Data & RLHF Introduction
**Topics:**
- Data mixing
- Rewriting techniques
- Supervised Fine-Tuning (SFT)
- Introduction to RLHF

**Tasks:**
- [ ] Watch Week 10 lectures
- [ ] Read: "Training Language Models to Follow Instructions"
- [ ] Complete Assignment 4 (Due this week)
- [ ] Start Assignment 5

**Key Papers:**
- "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)

**Deliverables:**
- Data processing pipeline
- Assignment 4 submission

---

## Path 2: Standard 15-Week Plan (Part-time)

### Weeks 1-2: Foundation
- Week 0-1 content
- Tokenization
- Basic PyTorch
- **Deliverable**: Tokenizer implementation

### Weeks 3-4: Transformers
- Week 2-3 content
- Architecture implementation
- Assignment 1
- **Deliverable**: Working Transformer

### Weeks 5-7: Systems
- Week 4-6 content
- Flash Attention
- Assignment 2
- **Deliverable**: Optimized attention

### Weeks 8-10: Scaling
- Week 7-8 content
- Scaling laws
- Assignment 3
- **Deliverable**: Scaling experiments

### Weeks 11-13: Data
- Week 9-10 content
- Data engineering
- Assignment 4
- **Deliverable**: Data pipeline

### Weeks 14-15: Alignment
- Week 11+ content
- RLHF
- Assignment 5
- **Deliverable**: Aligned model

## Path 3: Extended 20-Week Plan (Self-paced)

For those with limited time (5-8 hours/week):

- Each "week" of course content takes 1.5-2 real weeks
- More time for paper reading and experiments
- Buffer time for debugging and review

## Daily Study Template

### Option 1: Morning Session (2-3 hours)
```
09:00 - 09:30: Review yesterday's notes
09:30 - 10:30: Watch lecture video
10:30 - 11:00: Read related papers
11:00 - 12:00: Code implementation
```

### Option 2: Evening Session (2-3 hours)
```
19:00 - 19:30: Review concepts
19:30 - 20:30: Watch lecture
20:30 - 21:00: Take notes
21:00 - 22:00: Coding practice
```

## Weekly Review Checklist

Every Sunday, review:
- [ ] Lectures watched and understood
- [ ] Papers read and summarized
- [ ] Code written and tested
- [ ] Questions posted on Slack/Ed
- [ ] Progress on current assignment
- [ ] Plan for next week

## Assignment Deadlines (Reference)

Based on 2026 Schedule:
- **Assignment 1 Due**: Week 6
- **Assignment 2 Due**: Week 10
- **Assignment 3 Due**: Week 12
- **Assignment 4 Due**: Week 16
- **Assignment 5 Due**: Week 19

## Key Milestones

### Milestone 1: Working Transformer (End of Week 3)
- Can train a small language model
- Understands attention mechanism
- Completed Assignment 1

### Milestone 2: Efficient Training (End of Week 6)
- Implemented Flash Attention
- Understands distributed training
- Completed Assignment 2

### Milestone 3: Scaling Understanding (End of Week 8)
- Can predict training costs
- Understands scaling laws
- Completed Assignment 3

### Milestone 4: Data Mastery (End of Week 10)
- Can build data pipelines
- Understands data quality
- Completed Assignment 4

### Milestone 5: Full Stack (End of Week 12+)
- Understands alignment
- Can train aligned models
- Completed Assignment 5

## Pre-Course Preparation (2-4 weeks before)

### Math Review
- [ ] Linear Algebra (matrix operations, eigenvalues)
- [ ] Calculus (gradients, partial derivatives)
- [ ] Probability (distributions, Bayes' rule)
- [ ] Statistics (mean, variance, hypothesis testing)

### Programming Review
- [ ] Python proficiency
- [ ] PyTorch basics
- [ ] NumPy operations
- [ ] Git workflow

### Prerequisites Courses
- [ ] CS109 or equivalent (Probability)
- [ ] MATH51 or equivalent (Linear Algebra)
- [ ] CS221 or equivalent (ML basics)

## Resource Allocation

### Time Breakdown per Week
- Lectures: 3-4 hours
- Reading papers: 3-4 hours
- Coding: 6-8 hours
- Debugging: 2-3 hours
- Community/Office hours: 1-2 hours

### Priority Matrix

**High Priority (Must Do)**
- Complete all 5 assignments
- Implement core algorithms yourself
- Understand attention mechanism deeply

**Medium Priority (Should Do)**
- Read all required papers
- Participate in discussions
- Complete optional challenges

**Low Priority (Nice to Have)**
- Extra credit problems
- Extended experiments
- Side projects

## Progress Tracking Template

Create a file `my-progress.md`:

```markdown
# My CS336 Progress

## Week 1
- [x] Watched lectures
- [x] Read tokenization papers
- [ ] Implemented tokenizer (in progress)
- Notes: [link to notes]
- Time spent: 12 hours

## Week 2
...

## Assignments
- [ ] Assignment 1 (Due: XX/XX)
- [ ] Assignment 2 (Due: XX/XX)
...
```

## Tips for Success

1. **Start Early**: Don't wait until assignment due dates
2. **Ask Questions**: Use Slack and office hours
3. **Collaborate**: Form study groups (within honor code)
4. **Debug Systematically**: Use logging and profilers
5. **Save Checkpoints**: Always save model checkpoints
6. **Read Carefully**: Assignment specs are detailed
7. **Test Incrementally**: Test each component before full integration

## Warning Signs (When to Seek Help)

- Stuck on same bug for >3 days
- Falling >1 week behind schedule
- Not understanding lecture concepts
- Unable to complete basic exercises

## Alternative Resources

If stuck on concepts:
- Stanford CS224N (NLP with Deep Learning)
- Stanford CS230 (Deep Learning)
- 3Blue1Brown (Linear Algebra & Calculus)
- Andrej Karpathy's Neural Networks: Zero to Hero

## Final Checklist

Before starting:
- [ ] Environment set up
- [ ] GPU access confirmed
- [ ] Joined course Slack
- [ ] Read full syllabus
- [ ] Created study schedule
- [ ] Set up progress tracking

Good luck! 🚀
