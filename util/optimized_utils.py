"""
Optimized Utilities for G-LED
==============================

Provides:
- Model checkpointing and loading
- Logging utilities
- Metric tracking
- GPU monitoring
- Training visualization
- Reproducibility utilities
"""

import os
import json
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path
import time
from datetime import datetime
import subprocess


def set_random_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Use deterministic algorithms (slower but fully reproducible)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # Use deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Fallback for older PyTorch versions
            torch.set_deterministic(True)
    else:
        # Enable benchmarking for performance (non-deterministic)
        torch.backends.cudnn.benchmark = True
    
    print(f"✅ Random seed set to {seed} (deterministic={deterministic})")


class ModelCheckpointer:
    """
    Advanced model checkpointing with best model tracking.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
        save_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        
        self.best_metric = float('inf')
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        extra_state: Optional[Dict] = None
    ):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint.update(extra_state)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoint_history.append(checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model if applicable
        if self.save_best and 'loss' in metrics:
            if metrics['loss'] < self.best_metric:
                self.best_metric = metrics['loss']
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"⭐ New best model! Loss: {metrics['loss']:.6f}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only last N"""
        if len(self.checkpoint_history) > self.keep_last_n:
            to_remove = self.checkpoint_history[:-self.keep_last_n]
            for path in to_remove:
                if path.exists() and 'best' not in path.name:
                    path.unlink()
            self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint to load (or 'best'/'latest')
        
        Returns:
            Dictionary with epoch, metrics, etc.
        """
        if checkpoint_path is None or checkpoint_path == 'latest':
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]
        elif checkpoint_path == 'best':
            checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            if not checkpoint_path.exists():
                raise FileNotFoundError("Best model not found")
        else:
            checkpoint_path = Path(checkpoint_path)
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint


class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_utilization': [],
            'gpu_memory': []
        }
        
        self.epochs = []
    
    def update(self, epoch: int, **kwargs):
        """Update metrics for an epoch"""
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def save(self):
        """Save metrics to JSON"""
        metrics_file = self.log_dir / 'metrics.json'
        data = {
            'epochs': self.epochs,
            **self.metrics
        }
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot(self, save_path: Optional[str] = None):
        """Plot training metrics"""
        if not self.epochs:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Loss
        if self.metrics['train_loss']:
            axes[0, 0].plot(self.epochs, self.metrics['train_loss'], label='Train Loss')
        if self.metrics['valid_loss']:
            axes[0, 0].plot(self.epochs, self.metrics['valid_loss'], label='Valid Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning Rate
        if self.metrics['learning_rate']:
            axes[0, 1].plot(self.epochs, self.metrics['learning_rate'])
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
        
        # Epoch Time
        if self.metrics['epoch_time']:
            axes[1, 0].plot(self.epochs, self.metrics['epoch_time'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (s)')
            axes[1, 0].grid(True)
        
        # GPU Utilization
        if self.metrics['gpu_utilization']:
            axes[1, 1].plot(self.epochs, self.metrics['gpu_utilization'], label='GPU Util (%)')
        if self.metrics['gpu_memory']:
            ax2 = axes[1, 1].twinx()
            ax2.plot(self.epochs, self.metrics['gpu_memory'], 'r-', label='GPU Mem (MB)')
            ax2.set_ylabel('Memory (MB)', color='r')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"📊 Metrics plot saved to {save_path}")


class GPUMonitor:
    """Monitor GPU usage during training"""
    
    @staticmethod
    def get_gpu_stats() -> Dict[str, float]:
        """Get current GPU stats"""
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode()
            gpu_util, mem_used = output.strip().split(',')
            return {
                'utilization': float(gpu_util.strip()),
                'memory_mb': float(mem_used.strip())
            }
        except:
            return {'utilization': 0.0, 'memory_mb': 0.0}
    
    @staticmethod
    def print_gpu_stats():
        """Print GPU stats to console"""
        stats = GPUMonitor.get_gpu_stats()
        print(f"🖥️  GPU: {stats['utilization']:.1f}% utilized, "
              f"{stats['memory_mb']:.0f} MB used")


class TrainingLogger:
    """
    Comprehensive training logger combining all utilities.
    """
    
    def __init__(self, config):
        self.config = config
        self.checkpointer = ModelCheckpointer(
            config.checkpoint_dir,
            keep_last_n=config.training.keep_last_n_checkpoints
        )
        self.metrics_tracker = MetricsTracker(config.log_dir)
        self.gpu_monitor = GPUMonitor()
        
        self.epoch_start_time = None
    
    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch"""
        self.epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{self.config.training.epoch_num}")
        print(f"{'='*70}")
    
    def on_epoch_end(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        train_loss: float,
        valid_loss: Optional[float] = None,
        extra_metrics: Optional[Dict] = None
    ):
        """Called at the end of each epoch"""
        epoch_time = time.time() - self.epoch_start_time
        gpu_stats = self.gpu_monitor.get_gpu_stats()
        
        # Get learning rate
        lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics = {
            'train_loss': train_loss,
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'gpu_utilization': gpu_stats['utilization'],
            'gpu_memory': gpu_stats['memory_mb']
        }
        
        if valid_loss is not None:
            metrics['valid_loss'] = valid_loss
        
        if extra_metrics:
            metrics.update(extra_metrics)
        
        self.metrics_tracker.update(epoch, **metrics)
        
        # Save checkpoint
        if epoch % self.config.training.save_every == 0:
            self.checkpointer.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={'loss': valid_loss if valid_loss is not None else train_loss}
            )
        
        # Print summary
        print(f"\n📈 Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        if valid_loss is not None:
            print(f"  Valid Loss: {valid_loss:.6f}")
        print(f"  Learning Rate: {lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s ({3600/epoch_time:.1f} epochs/hour)")
        self.gpu_monitor.print_gpu_stats()
        print(f"{'='*70}\n")
        
        # Save metrics
        self.metrics_tracker.save()
        
        # Plot every 10 epochs
        if epoch % 10 == 0:
            self.metrics_tracker.plot()
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint='best'):
        """Load a checkpoint"""
        return self.checkpointer.load_checkpoint(
            model, optimizer, scheduler, checkpoint
        )


def save_model_simple(model, filepath):
    """Simple model saving"""
    torch.save(model.state_dict(), filepath)
    print(f"💾 Model saved to {filepath}")


def load_model_simple(model, filepath):
    """Simple model loading"""
    model.load_state_dict(torch.load(filepath))
    print(f"📂 Model loaded from {filepath}")
    return model


if __name__ == '__main__':
    print("Testing Utilities\n")
    
    # Test metrics tracker
    print("Testing MetricsTracker...")
    tracker = MetricsTracker('test_logs')
    for epoch in range(10):
        tracker.update(
            epoch=epoch,
            train_loss=1.0 / (epoch + 1),
            valid_loss=1.2 / (epoch + 1),
            learning_rate=1e-4 * 0.95**epoch
        )
    tracker.save()
    tracker.plot()
    print("✅ MetricsTracker works!\n")
    
    # Test GPU monitor
    print("Testing GPUMonitor...")
    GPUMonitor.print_gpu_stats()
    print("✅ GPUMonitor works!\n")
