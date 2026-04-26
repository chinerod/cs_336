# Triton 速查表

## 📚 基础语法

### Kernel定义
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, n):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)
```

### 启动Kernel
```python
import triton

grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
my_kernel[grid](x, y, n, BLOCK=256)
```

---

## 📚 常用操作

### 内存操作
```python
# 加载
tl.load(ptr, mask=mask)

# 存储
tl.store(ptr, value, mask=mask)

# 原子操作
tl.atomic_add(ptr, value)
```

### 数学操作
```python
# 基本运算
x + y, x - y, x * y, x / y

# 三角函数
tl.sin(x), tl.cos(x), tl.exp(x)

# 其他
tl.log(x), tl.sqrt(x), tl.maximum(x, y)
```

### 矩阵操作
```python
# 点积
tl.dot(A, B)

# 转置
tl.trans(A)
```

---

## 📚 高级特性

### Autotune
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}),
        triton.Config({'BLOCK': 256}),
    ],
    key=['n']
)
@triton.jit
def kernel(...):
    pass
```

---

*Triton速查表*
