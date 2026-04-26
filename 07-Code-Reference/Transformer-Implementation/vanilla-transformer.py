"""
Transformer Implementation Reference
CS336: Language Modeling from Scratch

This module provides clean, well-documented implementations of core
transformer components used in CS336.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need"

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability

    Reference:
        Vaswani et al. "Attention Is All You Need" (2017)
    """
    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention

        Args:
            Q: Query tensor [batch, n_heads, seq_len, d_k]
            K: Key tensor [batch, n_heads, seq_len, d_k]
            V: Value tensor [batch, n_heads, seq_len, d_k]
            mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            output: Attention output
            attention_weights: Attention weights for visualization
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if is_causal:
            # Causal (autoregressive) masking
            seq_len = Q.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device),
                diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: Optional mask
            is_causal: Apply causal masking

        Returns:
            output: [batch, seq_len, d_model]
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask, is_causal)

        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW1 + b1)W2 + b2

    Reference:
        Vaswani et al. "Attention Is All You Need" (2017)
    """
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Complete Transformer block with pre-norm architecture

    Pre-norm: LayerNorm -> Sub-layer -> Residual
    (More stable than post-norm for deep transformers)
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
            is_causal: Apply causal masking
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Self-attention with residual
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask, is_causal)
        x = x + self.dropout1(attn_output)

        # Feed-forward with residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)

        return x


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model

    Architecture:
    - Token embeddings
    - Positional encodings
    - N transformer blocks
    - Output projection to vocab

    Reference:
        GPT-style autoregressive language model
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and output
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token indices
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Create position indices
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Apply transformer blocks with causal masking
        for block in self.blocks:
            x = block(x, is_causal=True)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively

        Args:
            input_ids: [batch, seq_len] starting tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
        Returns:
            generated: [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids)

            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Example usage
if __name__ == "__main__":
    # Create model
    model = TransformerLM(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_ff=1024
    )

    # Forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
