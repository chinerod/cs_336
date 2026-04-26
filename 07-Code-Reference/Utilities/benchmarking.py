# Benchmarking Tools

```python
import torch
import time
from typing import Callable, List
import statistics


def benchmark(func: Callable, *args, num_runs: int = 100, warmup: int = 10):
    """
    基准测试函数

    Args:
        func: 要测试的函数
        args: 函数参数
        num_runs: 运行次数
        warmup: 预热次数

    Returns:
        dict: 包含mean, median, std等统计信息
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }


def compare_implementations(impls: List[tuple], *args, num_runs: int = 100):
    """
    对比多个实现

    Args:
        impls: [(name, func), ...]
        args: 函数参数
        num_runs: 运行次数
    """
    results = {}
    for name, func in impls:
        results[name] = benchmark(func, *args, num_runs=num_runs)

    print(f"{'Implementation':<20} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std':<10}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<20} {stats['mean']:<12.3f} {stats['median']:<12.3f} {stats['std']:<10.3f}")

    return results


def measure_flops(model, input_shape, device='cuda'):
    """
    估算FLOPs

    Args:
        model: PyTorch模型
        input_shape: 输入形状
        device: 设备

    Returns:
        dict: FLOPs统计
    """
    # 简化的FLOPs估算
    # 实际应使用thop或fvcore

    x = torch.randn(*input_shape).to(device)
    model = model.to(device)

    # Forward pass time
    start = time.perf_counter()
    y = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    forward_time = time.perf_counter() - start

    return {
        'forward_time_ms': forward_time * 1000,
        'note': 'Use thop or fvcore for accurate FLOPs'
    }


class Benchmark:
    """基准测试类"""

    def __init__(self, name: str):
        self.name = name
        self.results = []

    def run(self, func: Callable, *args, num_runs: int = 100):
        """运行测试"""
        stats = benchmark(func, *args, num_runs=num_runs)
        self.results.append({
            'name': self.name,
            'stats': stats
        })
        return stats

    def report(self):
        """生成报告"""
        print(f"\nBenchmark: {self.name}")
        print("-" * 40)
        for result in self.results:
            stats = result['stats']
            print(f"Mean: {stats['mean']:.3f} ms")
            print(f"Median: {stats['median']:.3f} ms")
            print(f"Std: {stats['std']:.3f} ms")
            print(f"Min/Max: {stats['min']:.3f} / {stats['max']:.3f} ms")
```

---

*基准测试工具*
