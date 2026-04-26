# PyTorch 速查表（中文版）

## 📚 快速参考

### 导入
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

---

## 🔧 Tensor操作

### 创建Tensor
```python
# 从数据创建
x = torch.tensor([1, 2, 3])
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# 特殊Tensor
x = torch.zeros(2, 3)          # 全0
x = torch.ones(2, 3)           # 全1
x = torch.eye(3)               # 单位矩阵
x = torch.full((2, 3), 7)      # 填充特定值
x = torch.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)    # [0, 0.25, 0.5, 0.75, 1]

# 随机Tensor
x = torch.rand(2, 3)           # 均匀分布 [0, 1)
x = torch.randn(2, 3)          # 标准正态分布
x = torch.randint(0, 10, (2, 3))  # 整数随机 [0, 10)
```

### Tensor属性
```python
x.shape          # 形状 (torch.Size)
 x.size()        # 同上
x.dim()          # 维度数
x.numel()        # 元素总数
x.dtype          # 数据类型
x.device         # 设备 (cpu/cuda)
```

### 形状操作
```python
x.view(3, 4)           # 改变形状（共享内存）
x.reshape(3, 4)        # 改变形状（可能复制）
x.unsqueeze(0)         # 在dim=0添加维度
x.squeeze()            # 移除所有size=1的维度
x.squeeze(0)           # 移除指定维度
x.flatten()            # 展平
x.transpose(0, 1)      # 交换维度
x.permute(2, 0, 1)     # 重排维度
x.expand(3, 4)         # 广播扩展（不复制）
```

### 索引和切片
```python
x[0]                   # 第0个元素
x[0, :]                # 第0行
x[:, 0]                # 第0列
x[0:3]                 # 切片
x[x > 0]               # 布尔索引
torch.where(x > 0, x, y)  # 条件选择
```

### 数学运算
```python
# 基本运算（支持广播）
x + y
x - y
x * y                  # 逐元素乘
x / y
x @ y                  # 矩阵乘法
x ** 2                 # 幂运算

# 逐元素函数
torch.abs(x)
torch.exp(x)
torch.log(x)
torch.sqrt(x)
torch.sigmoid(x)
torch.tanh(x)
torch.relu(x)

# 规约操作
x.sum()                # 求和
x.sum(dim=0)           # 按维度求和
x.mean()               # 平均
x.max()                # 最大值
x.min()                # 最小值
x.argmax()             # 最大值索引
x.std()                # 标准差
```

---

## 💾 设备管理

### CPU/GPU切换
```python
# 创建时指定设备
x = torch.tensor([1, 2, 3], device='cuda')
x = torch.randn(2, 3, device='cuda:0')

# 移动到设备
x = x.to('cuda')       # 或 x.cuda()
x = x.to('cpu')        # 或 x.cpu()
x = x.to(device)       # device可以是字符串或torch.device

# 检查GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
else:
    device = torch.device('cpu')

# 获取设备上的tensor
x = torch.randn(2, 3)
x = x.to(device)
```

---

## 🔀 自动微分 (Autograd)

### 基本使用
```python
# 创建需要梯度的tensor
x = torch.tensor([1.0, 2.0], requires_grad=True)

# 前向计算
y = x ** 2
z = y.sum()

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # tensor([2., 4.])
```

### 梯度操作
```python
# 清零梯度
x.grad.zero_()
optimizer.zero_grad()  # 更常用

# 阻止梯度计算
with torch.no_grad():
    y = model(x)

# 或者使用detach()
y = x.detach()

# 查看是否需要梯度
x.requires_grad

# 设置不需要梯度
x.requires_grad_(False)
```

### 梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

---

## 🏗️ 神经网络 (nn.Module)

### 定义模型
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 实例化
model = MyModel()
model = model.to(device)
```

### 常用层

#### 线性层
```python
nn.Linear(in_features=784, out_features=256, bias=True)
```

#### 卷积层
```python
nn.Conv1d(in_channels, out_channels, kernel_size)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.Conv3d(in_channels, out_channels, kernel_size)

# 转置卷积
nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
```

#### 池化层
```python
nn.MaxPool2d(kernel_size, stride)
nn.AvgPool2d(kernel_size, stride)
nn.AdaptiveAvgPool2d(output_size)
```

#### 归一化层
```python
nn.BatchNorm1d(num_features)
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)
nn.GroupNorm(num_groups, num_channels)
```

#### 激活函数
```python
nn.ReLU()
nn.LeakyReLU(negative_slope=0.01)
nn.GELU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.LogSoftmax(dim=1)
```

#### Dropout
```python
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)
```

#### Embedding
```python
nn.Embedding(num_embeddings=10000, embedding_dim=256)
```

### 容器
```python
nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
```

---

## ⚙️ 损失函数

### 分类
```python
nn.CrossEntropyLoss()           # 多分类（Softmax + NLL）
nn.BCELoss()                    # 二分类（需要sigmoid）
nn.BCEWithLogitsLoss()          # 二分类（自带sigmoid）
nn.NLLLoss()                    # 负对数似然
```

### 回归
```python
nn.MSELoss()                    # 均方误差
nn.L1Loss()                     # L1损失
nn.SmoothL1Loss()               # 平滑L1
```

### 其他
```python
nn.KLDivLoss()                  # KL散度
nn.CosineEmbeddingLoss()        # 余弦相似度
nn.TripletMarginLoss()          # 三元组损失
```

---

## 🔧 优化器

### 常用优化器
```python
# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

### 优化器操作
```python
optimizer.zero_grad()      # 清零梯度
optimizer.step()           # 更新参数
optimizer.state_dict()     # 保存优化器状态
optimizer.load_state_dict(state_dict)  # 加载
```

### 学习率调度
```python
# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Warmup + Cosine
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)

# OneCycle
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=100, epochs=10)

# 每个epoch调用
scheduler.step()
```

---

## 🔄 数据加载

### Dataset
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### DataLoader
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# 迭代
for batch_data, batch_labels in dataloader:
    # 训练
```

### 常用Dataset
```python
from torchvision import datasets, transforms

# 图像
 dataset = datasets.CIFAR10(root='./data', train=True, download=True)

# 变换
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## 💾 保存和加载

### 保存模型
```python
# 保存整个模型（不推荐）
torch.save(model, 'model.pth')

# 保存状态字典（推荐）
torch.save(model.state_dict(), 'model.pth')

# 保存检查点
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

### 加载模型
```python
# 加载整个模型
model = torch.load('model.pth')

# 加载状态字典
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 🎓 训练模式

### 训练/评估
```python
model.train()   # 训练模式（启用dropout, batchnorm统计）
model.eval()    # 评估模式（关闭dropout, 使用running statistics）

# 推理时不计算梯度
with torch.no_grad():
    output = model(input)
```

### 训练循环模板
```python
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🚀 高级操作

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    data, target = data.to(device), target.to(device)

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 分布式训练
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend='nccl')

# 包装模型
model = DDP(model, device_ids=[local_rank])

# 同步
 dist.barrier()
```

### TorchScript
```python
# 导出
scripted_model = torch.jit.script(model)
scripted_model.save('model.pt')

# 加载
model = torch.jit.load('model.pt')
```

---

## 🐛 调试技巧

### 查看Tensor信息
```python
print(x.shape)
print(x.dtype)
print(x.device)
print(x.min(), x.max(), x.mean())
```

### 检查NaN/Inf
```python
torch.isnan(x).any()
torch.isinf(x).any()
torch.isfinite(x).all()
```

### 梯度检查
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### 显存检查
```python
# 当前显存使用
torch.cuda.memory_allocated() / 1e9  # GB

# 最大显存使用
torch.cuda.max_memory_allocated() / 1e9

# 显存摘要
print(torch.cuda.memory_summary())
```

---

## 📊 常用函数速查

| 操作 | 函数 |
|------|------|
| 矩阵乘法 | `torch.matmul`, `@` |
| 逐元素乘 | `torch.mul`, `*` |
| 拼接 | `torch.cat`, `torch.stack` |
| 分割 | `torch.split`, `torch.chunk` |
| 转置 | `tensor.T`, `torch.transpose` |
| 类型转换 | `tensor.float()`, `tensor.long()` |
| 比较 | `torch.eq`, `torch.gt`, `torch.lt` |
| 随机 | `torch.rand`, `torch.randn` |
| 排序 | `torch.sort`, `torch.topk` |
| 掩码 | `torch.masked_fill`, `torch.where` |

---

## 💡 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: CUDA out of memory` | 显存不足 | 减小batch size，使用torch.cuda.empty_cache() |
| `RuntimeError: expected ... got ...` | 数据类型不匹配 | 使用tensor.float()或tensor.long()转换 |
| `RuntimeError: shape mismatch` | 形状不匹配 | 检查tensor.shape |
| `AttributeError: 'NoneType' ...` | 忘记zero_grad或forward | 检查流程 |
| `RuntimeError: grad can be implicitly ...` | inplace操作 | 避免x += 1，用x = x + 1 |

---

**收藏这份速查表，随时查阅！** 🚀

