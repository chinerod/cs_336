# PyTorch Cheat Sheet for CS336

## Tensor Operations

### Creating Tensors
```python
import torch

# Basic creation
x = torch.tensor([1, 2, 3])
x = torch.zeros(2, 3)
x = torch.ones(2, 3)
x = torch.eye(3)  # Identity matrix
x = torch.rand(2, 3)
x = torch.randn(2, 3)  # Normal distribution

# GPU tensors
x = torch.zeros(2, 3, device='cuda')
x = x.cuda()
x = x.cpu()

# Shapes
x.shape        # torch.Size([2, 3])
x.size()       # Same as above
x.dim()        # 2
x.numel()      # 6 (total elements)

# Type conversion
x = x.float()
x = x.long()
x = x.half()   # FP16
x = x.bfloat16()  # BF16
```

### Tensor Manipulation
```python
# Reshape
x = x.view(3, 2)      # Contiguous
x = x.reshape(3, 2)   # Any layout
x = x.flatten()
x = x.unsqueeze(0)    # Add dimension
x = x.squeeze(0)      # Remove dimension

# Indexing
x[0]
x[0:2]
x[:, 0]
x[x > 0.5]            # Boolean indexing

# Concatenate
y = torch.cat([x1, x2], dim=0)
y = torch.stack([x1, x2], dim=0)

# Split
chunks = torch.chunk(x, 4, dim=0)
splits = torch.split(x, 2, dim=0)
```

## Neural Network Layers

### Linear Layers
```python
import torch.nn as nn

# Linear layer
linear = nn.Linear(512, 1024)
x = linear(input)  # input: [batch, 512], output: [batch, 1024]

# Embedding
emb = nn.Embedding(10000, 512)  # vocab_size, dim
x = emb(input_ids)  # input: [batch, seq], output: [batch, seq, 512]

# Layer Norm
ln = nn.LayerNorm(512)
x = ln(x)  # input: [..., 512]

# Dropout
dropout = nn.Dropout(0.1)
x = dropout(x)
```

### Activation Functions
```python
import torch.nn.functional as F

# Common activations
x = F.relu(x)
x = F.gelu(x)        # Recommended for transformers
x = F.sigmoid(x)
x = F.tanh(x)
x = F.softmax(x, dim=-1)
x = F.log_softmax(x, dim=-1)

# GELU variants
x = F.gelu(x, approximate='tanh')
```

### Loss Functions
```python
# Cross entropy (classification)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

# MSE (regression)
criterion = nn.MSELoss()
loss = criterion(predictions, targets)

# Custom loss
loss = F.cross_entropy(logits, targets, ignore_index=-100)
```

## Optimizers

```python
# AdamW (recommended)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

scheduler = CosineAnnealingLR(optimizer, T_max=1000)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000)

# Training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training step
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

## Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
import os
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Synchronize
dist.barrier()

# All reduce (manual)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

## Checkpointing

```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Memory Management

```python
# Clear cache
torch.cuda.empty_cache()

# Memory stats
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.get_device_properties(0).total_memory

# Automatic mixed precision
with torch.cuda.amp.autocast():
    pass

# Gradient checkpointing
model.gradient_checkpointing_enable()
```

## einops (Important for CS336)

```python
from einops import rearrange, reduce, repeat

# Rearrange
x = rearrange(x, 'b h w c -> b c h w')
x = rearrange(x, 'b s (h d) -> b h s d', h=n_heads)

# Reduce
x = reduce(x, 'b c h w -> b c', 'mean')
x = reduce(x, 'b h s d -> b s d', 'max')

# Repeat
x = repeat(x, 'b c h w -> b c (h 2) (w 2)')
x = repeat(x, 'b s d -> b (s 2) d')

# Common patterns for transformers
q = rearrange(q, 'b s (h d) -> b h s d', h=n_heads)
k = rearrange(k, 'b s (h d) -> b h s d', h=n_heads)
v = rearrange(v, 'b s (h d) -> b h s d', h=n_heads)

# Merge heads back
out = rearrange(out, 'b h s d -> b s (h d)')
```

## Common Patterns

### Attention Mechanism
```python
# Q, K, V computation
Q = W_q(x)  # [batch, seq, dim]
K = W_k(x)
V = W_v(x)

# Split heads
Q = rearrange(Q, 'b s (h d) -> b h s d', h=n_heads)
K = rearrange(K, 'b s (h d) -> b h s d', h=n_heads)
V = rearrange(V, 'b s (h d) -> b h s d', h=n_heads)

# Attention scores
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
attn = F.softmax(scores, dim=-1)
out = torch.matmul(attn, V)

# Merge heads
out = rearrange(out, 'b h s d -> b s (h d)')
```

### Gradient Accumulation
```python
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    with autocast():
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### DataLoader
```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(data)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
)
```

## Debugging Tips

```python
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")

# Detect anomalies
torch.autograd.set_detect_anomaly(True)

# Profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model(input)
print(prof.key_averages().table())

# Print model summary
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Performance Tips

```python
# Enable TF32 (Ampere GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Benchmark mode
torch.backends.cudnn.benchmark = True

# Deterministic (for reproducibility)
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

# Compile model (PyTorch 2.0+)
model = torch.compile(model)
```

## Common Errors & Fixes

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size, use gradient accumulation |
| `Expected tensor on GPU` | Call `.cuda()` or use `device='cuda'` |
| `Dimension mismatch` | Check tensor shapes with `.shape` |
| `RuntimeError: CUDA error` | Clear cache: `torch.cuda.empty_cache()` |
| `Gradient is None` | Check requires_grad, call backward after forward |
