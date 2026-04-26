"""
Flash Attention Implementation Reference
CS336 Assignment 2: Systems

This module provides Flash Attention implementations:
1. Baseline attention (for comparison)
2. Memory-efficient attention (naive)
3. Flash Attention v1 (tiling algorithm)
4. Flash Attention v2 (improved parallelism)

Reference:
- FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)
- FlashAttention-2: Faster Attention with Better Parallelism (Dao, 2023)
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def baseline_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False
) -> torch.Tensor:
    """
    Baseline attention implementation (O(N^2) memory)

    Args:
        Q: [batch, n_heads, seq_len, head_dim]
        K: [batch, n_heads, seq_len, head_dim]
        V: [batch, n_heads, seq_len, head_dim]
        causal: Whether to apply causal masking

    Returns:
        output: [batch, n_heads, seq_len, head_dim]
    """
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

    if causal:
        # Causal mask
        seq_len = Q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, V)

    return output


def memory_efficient_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    chunk_size: int = 1024
) -> torch.Tensor:
    """
    Memory-efficient attention using chunking

    Process output in chunks to reduce memory usage

    Args:
        Q, K, V: Attention tensors
        causal: Causal masking
        chunk_size: Size of chunks for processing

    Returns:
        output: Attention output
    """
    batch_size, n_heads, seq_len, head_dim = Q.shape
    output = torch.zeros_like(Q)

    # Process in chunks
    for i in range(0, seq_len, chunk_size):
        j = min(i + chunk_size, seq_len)
        Qi = Q[:, :, i:j, :]

        # Compute attention for this chunk
        scores = torch.matmul(Qi, K.transpose(-2, -1)) / math.sqrt(head_dim)

        if causal:
            # Apply causal mask per chunk
            mask = torch.triu(
                torch.ones(j - i, seq_len, device=Q.device),
                diagonal=i+1
            ).bool()
            scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output[:, :, i:j, :] = torch.matmul(attn_weights, V)

    return output


class FlashAttentionV1:
    """
    Flash Attention v1 - Tiling algorithm

    Key insight: Compute attention in tiles to fit in SRAM

    SRAM (fast, small) -> Compute softmax incrementally -> HBM (slow, large)

    Reference:
        "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    """

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Flash Attention forward pass

        Args:
            Q, K, V: [batch, n_heads, seq_len, head_dim]
            causal: Causal masking

        Returns:
            O: [batch, n_heads, seq_len, head_dim]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        block_size = self.block_size

        # Output tensor
        O = torch.zeros_like(Q)

        # Scaling factor
        scale = 1.0 / math.sqrt(head_dim)

        # Tile over sequence length
        num_blocks = (seq_len + block_size - 1) // block_size

        for i in range(num_blocks):
            # Define tile range
            tile_start = i * block_size
            tile_end = min((i + 1) * block_size, seq_len)
            tile_len = tile_end - tile_start

            # Load Q tile into SRAM
            Qi = Q[:, :, tile_start:tile_end, :]

            # Initialize statistics
            mi = torch.full((batch_size, n_heads, tile_len), float('-inf'), device=Q.device)
            li = torch.zeros((batch_size, n_heads, tile_len), device=Q.device)
            Oi = torch.zeros((batch_size, n_heads, tile_len, head_dim), device=Q.device)

            # Iterate over K, V tiles
            for j in range(num_blocks if not causal else i + 1):
                kv_start = j * block_size
                kv_end = min((j + 1) * block_size, seq_len)

                # Load Kj, Vj into SRAM
                Kj = K[:, :, kv_start:kv_end, :]
                Vj = V[:, :, kv_start:kv_end, :]

                # Compute Sij = Qi * Kj^T
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

                if causal and j == i:
                    # Apply causal mask within tile
                    mask = torch.triu(
                        torch.ones(tile_len, kv_end - kv_start, device=Q.device),
                        diagonal=1
                    ).bool()
                    Sij.masked_fill_(mask, float('-inf'))

                # Compute softmax statistics
                mij = torch.max(Sij, dim=-1, keepdim=True).values
                Pij = torch.exp(Sij - mij)
                lij = torch.sum(Pij, dim=-1, keepdim=True)

                # Update running statistics
                mi_new = torch.max(torch.cat([mi.unsqueeze(-1), mij], dim=-1), dim=-1).values
                li_new = torch.exp(mi - mi_new) * li + torch.exp(mij.squeeze(-1) - mi_new) * lij.squeeze(-1)

                # Update output
                Oi = (li.unsqueeze(-1) * torch.exp(mi - mi_new).unsqueeze(-1) * Oi +
                      torch.exp(mij - mi_new.unsqueeze(-1)) * torch.matmul(Pij, Vj)) / li_new.unsqueeze(-1)

                mi = mi_new
                li = li_new

            # Store output
            O[:, :, tile_start:tile_end, :] = Oi

        return O


class FlashAttentionV2:
    """
    Flash Attention v2 - Improved parallelism

    Improvements over v1:
    1. Better parallelism (reduces synchronization)
    2. Work partitioning (equal work per thread)
    3. Reduced non-matmul FLOPs

    Reference:
        "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
    """

    def __init__(self, block_size_q: int = 128, block_size_kv: int = 128):
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Flash Attention v2 forward pass

        Key difference: Separate row and column parallelism
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        block_q = self.block_size_q
        block_kv = self.block_size_kv

        O = torch.zeros_like(Q)
        scale = 1.0 / math.sqrt(head_dim)

        num_blocks_q = (seq_len + block_q - 1) // block_q

        for i in range(num_blocks_q):
            q_start = i * block_q
            q_end = min((i + 1) * block_q, seq_len)
            q_len = q_end - q_start

            Qi = Q[:, :, q_start:q_end, :]

            # Initialize row-wise statistics
            mi = torch.full((batch_size, n_heads, q_len), float('-inf'), device=Q.device)
            li = torch.zeros((batch_size, n_heads, q_len), device=Q.device)
            Oi = torch.zeros((batch_size, n_heads, q_len, head_dim), device=Q.device)

            # Determine KV range (causal)
            kv_end_block = (q_end + block_kv - 1) // block_kv if causal else (seq_len + block_kv - 1) // block_kv

            for j in range(kv_end_block):
                kv_start = j * block_kv
                kv_end = min((j + 1) * block_kv, seq_len)

                Kj = K[:, :, kv_start:kv_end, :]
                Vj = V[:, :, kv_start:kv_end, :]

                # Compute attention scores
                Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) * scale

                if causal and kv_end > q_start:
                    # Causal masking
                    row_idx = torch.arange(q_start, q_end, device=Q.device).view(-1, 1)
                    col_idx = torch.arange(kv_start, kv_end, device=Q.device).view(1, -1)
                    mask = col_idx > row_idx
                    Sij.masked_fill_(mask, float('-inf'))

                # Online softmax update
                mij = torch.max(Sij, dim=-1).values
                Pij = torch.exp(Sij - mij.unsqueeze(-1))
                lij = torch.sum(Pij, dim=-1)

                # Update running statistics
                mi_new = torch.maximum(mi, mij)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(mij - mi_new)

                li_new = alpha * li + beta * lij

                # Update output
                PV = torch.matmul(Pij, Vj)
                Oi = (Oi * alpha.unsqueeze(-1) + beta.unsqueeze(-1) * PV) / li_new.unsqueeze(-1)

                mi = mi_new
                li = li_new

            O[:, :, q_start:q_end, :] = Oi

        return O


class FlashAttentionTriton:
    """
    Placeholder for Triton-based Flash Attention

    In Assignment 2, you will implement this using Triton
    for maximum performance on GPU.

    Key concepts:
    - Triton kernel programming
    - Shared memory management
    - Warp-level primitives
    """

    def __init__(self):
        self.triton_available = False
        try:
            import triton
            import triton.language as tl
            self.triton_available = True
        except ImportError:
            print("Triton not available. Install with: pip install triton")

    def forward(self, Q, K, V, causal=False):
        """Placeholder - implement in Triton for Assignment 2"""
        raise NotImplementedError(
            "Implement this using Triton for Assignment 2!\n"
            "See: https://triton-lang.org/ for documentation"
        )


def benchmark_attention(
    seq_len: int = 2048,
    batch_size: int = 2,
    n_heads: int = 8,
    head_dim: int = 64,
    device: str = "cuda"
):
    """
    Benchmark different attention implementations

    Args:
        seq_len: Sequence length
        batch_size: Batch size
        n_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to run on
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Create tensors
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    print(f"Benchmarking attention with seq_len={seq_len}")
    print(f"QKV shape: {Q.shape}")
    print(f"Device: {device}")
    print()

    # Warm up CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark baseline
    import time

    def measure_time(fn, *args, **kwargs, num_runs=10):
        # Warm up
        for _ in range(3):
            fn(*args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(num_runs):
            fn(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
        end = time.time()

        return (end - start) / num_runs * 1000  # ms

    # Test implementations
    try:
        time_baseline = measure_time(baseline_attention, Q, K, V, causal=False)
        print(f"Baseline Attention: {time_baseline:.2f} ms")
    except Exception as e:
        print(f"Baseline failed: {e}")

    try:
        time_mem = measure_time(memory_efficient_attention, Q, K, V, causal=False)
        print(f"Memory Efficient: {time_mem:.2f} ms")
    except Exception as e:
        print(f"Memory efficient failed: {e}")

    # Memory usage
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        output = baseline_attention(Q, K, V)
        mem_baseline = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"\nMemory (baseline): {mem_baseline:.1f} MB")


# Example usage
if __name__ == "__main__":
    print("Flash Attention Implementation Reference")
    print("=" * 50)
    print()

    # Test dimensions
    batch_size = 2
    n_heads = 8
    seq_len = 1024
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Create test tensors
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    # Test implementations
    print("Testing implementations...")

    # Baseline
    output_baseline = baseline_attention(Q, K, V, causal=False)
    print(f"Baseline output shape: {output_baseline.shape}")

    # Flash Attention V1
    fa1 = FlashAttentionV1(block_size=128)
    output_fa1 = fa1.forward(Q, K, V, causal=False)
    print(f"Flash Attention V1 output shape: {output_fa1.shape}")

    # Check correctness
    diff = torch.abs(output_baseline - output_fa1).max()
    print(f"Max difference: {diff:.6f}")

    print("\n" + "=" * 50)
    print("Run benchmark_attention() for performance comparison")
