"""
Training Utilities for CS336
Includes: Distributed training, Mixed precision, Gradient accumulation, Checkpointing
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from typing import Optional, Dict, Any


class MixedPrecisionTrainer:
    """
    Automatic Mixed Precision (AMP) Trainer

    Uses FP16/BF16 for forward/backward passes, FP32 for optimizer steps
    Reduces memory usage and speeds up training on modern GPUs

    Reference:
        https://pytorch.org/docs/stable/amp.html
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        use_amp: bool = True,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_amp = use_amp and torch.cuda.is_available()
        self.device = device

        if self.use_amp:
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler = None

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        accumulation_steps: int = 1
    ) -> float:
        """
        Single training step with optional gradient accumulation

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target labels [batch, seq_len]
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            loss: Scalar loss value
        """
        batch_size = input_ids.size(0)

        with autocast(enabled=self.use_amp):
            # Forward pass
            logits = self.model(input_ids)

            # Compute loss (cross-entropy)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * accumulation_steps

    def optimizer_step(self):
        """Update weights with accumulated gradients"""
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        loss: float,
        path: str
    ):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint


class GradientAccumulator:
    """
    Gradient Accumulation

    Allows training with larger effective batch sizes by accumulating
    gradients over multiple forward/backward passes before updating weights

    Use case: When GPU memory is limited but you need large batch sizes
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def step(
        self,
        loss: torch.Tensor,
        is_last_step: bool
    ):
        """
        Accumulate gradients

        Args:
            loss: Loss tensor (should already be scaled by accumulation_steps)
            is_last_step: Whether this is the last accumulation step
        """
        # Backward
        loss.backward()

        self.current_step += 1

        # Update weights if accumulated enough
        if self.current_step % self.accumulation_steps == 0 or is_last_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0


class DistributedTrainer:
    """
    Distributed Data Parallel (DDP) Training

    Uses PyTorch DDP for multi-GPU training across multiple nodes

    Key features:
    - Automatic gradient synchronization
    - Scales to multiple nodes
    - Efficient communication backend

    Reference:
        https://pytorch.org/tutorials/beginner/dist_overview.html
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "nccl"
    ):
        """
        Args:
            model: PyTorch model
            backend: Communication backend (nccl for GPU, gloo for CPU)
        """
        if not dist.is_initialized():
            raise RuntimeError("Distributed training not initialized. "
                             "Call torch.distributed.init_process_group() first")

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()

        # Move model to correct device
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        model = model.to(self.device)

        # Wrap with DDP
        self.model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank
        )

        print(f"Initialized DDP on rank {self.local_rank}/{self.world_size}")

    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)"""
        return self.local_rank == 0

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save checkpoint (only from main process)"""
        if self.is_main_process():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, path)

        # Wait for all processes
        dist.barrier()


def setup_distributed():
    """Setup distributed training environment"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )

        torch.cuda.set_device(local_rank)
        print(f"Initialized distributed training: rank={rank}, world_size={world_size}")
        return True
    return False


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class CheckpointManager:
    """
    Manage model checkpoints with automatic saving and loading

    Features:
    - Automatic checkpointing every N steps
    - Keep only N most recent checkpoints
    - Save best model based on validation loss
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        keep_last_n: int = 3,
        save_best: bool = True
    ):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.best_loss = float('inf')
        self.checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        loss: float,
        is_best: bool = False
    ):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        # Regular checkpoint
        path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{step}.pt"
        )
        torch.save(checkpoint, path)
        self.checkpoints.append(path)

        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_path = self.checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)

        # Best checkpoint
        if self.save_best and (is_best or loss < self.best_loss):
            self.best_loss = loss
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss {loss:.4f}")

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        if not self.checkpoints:
            # Try to find checkpoints in directory
            import glob
            pattern = os.path.join(self.checkpoint_dir, "checkpoint_*.pt")
            checkpoints = glob.glob(pattern)
            if not checkpoints:
                print("No checkpoints found")
                return None
            self.checkpoints = sorted(checkpoints)

        latest_path = self.checkpoints[-1]
        checkpoint = torch.load(latest_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded checkpoint from {latest_path}")
        return checkpoint


# Example usage
if __name__ == "__main__":
    from ..Transformer.vanilla_transformer import TransformerLM

    # Create model
    model = TransformerLM(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        n_layers=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Mixed precision trainer
    trainer = MixedPrecisionTrainer(model, optimizer, use_amp=True)

    print("Training utilities initialized")
    print(f"AMP enabled: {trainer.use_amp}")
