# 内存分析工具

"""
内存分析工具

用于分析PyTorch模型的内存使用情况
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import gc


def get_model_memory(model: nn.Module) -> Dict[str, float]:
    """
    获取模型内存占用

    Returns:
        dict: 内存统计（MB）
    """
    mem_params = sum([p.nelement() * p.element_size() for p in model.parameters()])
    mem_buffers = sum([b.nelement() * b.element_size() for b in model.buffers()])
    mem_total = mem_params + mem_buffers

    return {
        'params_mb': mem_params / 1024**2,
        'buffers_mb': mem_buffers / 1024**2,
        'total_mb': mem_total / 1024**2,
        'params_count': sum(p.nelement() for p in model.parameters()),
    }


def print_model_memory(model: nn.Module, detailed: bool = False):
    """打印模型内存信息"""
    mem_info = get_model_memory(model)

    print("=" * 50)
    print("Model Memory Usage")
    print("=" * 50)
    print(f"Total Memory: {mem_info['total_mb']:.2f} MB")
    print(f"Parameters:   {mem_info['params_mb']:.2f} MB")
    print(f"Buffers:      {mem_info['buffers_mb']:.2f} MB")
    print(f"Param Count:  {mem_info['params_count']:,}")

    if detailed:
        print("\nDetailed Breakdown:")
        for name, param in model.named_parameters():
            size_mb = param.nelement() * param.element_size() / 1024**2
            print(f"  {name}: {size_mb:.2f} MB")


def get_gpu_memory():
    """获取GPU内存信息"""
    if not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'free_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved,
    }


def print_gpu_memory():
    """打印GPU内存信息"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    mem_info = get_gpu_memory()
    props = torch.cuda.get_device_properties(0)

    print("=" * 50)
    print("GPU Memory Status")
    print("=" * 50)
    print(f"Device: {props.name}")
    print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"Allocated:    {mem_info['allocated_gb']:.2f} GB")
    print(f"Reserved:     {mem_info['reserved_gb']:.2f} GB")
    print(f"Max Allocated:{mem_info['max_allocated_gb']:.2f} GB")
    print(f"Free:         {mem_info['free_gb']:.2f} GB")


def memory_profiler(func):
    """内存分析装饰器"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        gc.collect()

        print(f"\n{'='*50}")
        print(f"Profiling: {func.__name__}")
        print(f"{'='*50}")

        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**3
            print(f"Memory before: {mem_before:.2f} GB")

        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**3
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Memory after:  {mem_after:.2f} GB")
            print(f"Peak memory:   {mem_peak:.2f} GB")
            print(f"Delta:         {mem_after - mem_before:.2f} GB")

        return result

    return wrapper


class MemoryTracker:
    """内存跟踪器"""

    def __init__(self):
        self.snapshots = []

    def snapshot(self, name: str = ""):
        """记录内存快照"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        snapshot = {
            'name': name,
            'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            'reserved': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        }

        self.snapshots.append(snapshot)

    def report(self):
        """生成报告"""
        print("\n" + "=" * 60)
        print("Memory Tracking Report")
        print("=" * 60)

        for i, snap in enumerate(self.snapshots):
            allocated_mb = snap['allocated'] / 1024**2
            reserved_mb = snap['reserved'] / 1024**2
            print(f"{i+1}. {snap['name']:<20} Allocated: {allocated_mb:8.2f} MB, Reserved: {reserved_mb:8.2f} MB")

        if len(self.snapshots) > 1:
            print("\nDeltas:")
            for i in range(1, len(self.snapshots)):
                prev = self.snapshots[i-1]
                curr = self.snapshots[i]
                delta = (curr['allocated'] - prev['allocated']) / 1024**2
                print(f"  {prev['name']} -> {curr['name']}: {delta:+.2f} MB")


# 使用示例
if __name__ == '__main__':
    # 测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(1024, 4096)
            self.linear2 = nn.Linear(4096, 1024)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    model = TestModel()

    # 打印模型内存
    print_model_memory(model, detailed=True)

    # 测试GPU内存
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(128, 1024).cuda()

        print_gpu_memory()

        # 跟踪内存
        tracker = MemoryTracker()
        tracker.snapshot("Before forward")

        y = model(x)
        tracker.snapshot("After forward")

        loss = y.sum()
        loss.backward()
        tracker.snapshot("After backward")

        tracker.report()

