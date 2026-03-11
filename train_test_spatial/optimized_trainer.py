"""
Optimized Diffusion Model Training
===================================

Features:
- Uses ImagenTrainer with optimizations enabled
- Multi-GPU support via Accelerate
- Comprehensive logging
- Resume capability
- Best model tracking
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
from typing import Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import ExperimentConfig
from util.optimized_utils import TrainingLogger
from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer


class OptimizedDiffusionTrainer:
    """
    High-performance trainer for diffusion model.
    
    Leverages ImagenTrainer's built-in optimizations:
    - Accelerate for multi-GPU
    - Mixed precision
    - EMA
    - Gradient accumulation
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        imagen: ElucidatedImagen,
        train_loader,
        valid_loader,
        seq_model_config: Optional[ExperimentConfig] = None,
        wandb_run=None
    ):
        self.config = config
        self.imagen = imagen
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.seq_model_config = seq_model_config
        self.wandb_run = wandb_run
        
        # Create ImagenTrainer with optimizations
        self.trainer = self._create_trainer()
        
        # Samplers (create once!)
        self.down_sampler = None
        self.up_sampler = None
        if seq_model_config is not None:
            self.down_sampler = nn.Upsample(
                size=seq_model_config.model.coarse_dim,
                mode=seq_model_config.model.coarse_mode
            ).to(config.training.device)
            
            self.up_sampler = nn.Upsample(
                size=[config.data.image_height, config.data.image_width],
                mode=seq_model_config.model.coarse_mode
            ).to(config.training.device)
        
        # Logging
        self.logger = TrainingLogger(config)
        
        print(f"\n✅ Diffusion Trainer initialized")
        print(f"   Image Size: {config.data.image_height}×{config.data.image_width}")
        print(f"   UNet Dim: {config.model.unet_dim}")
        print(f"   Mixed Precision: {config.training.use_amp}")
        print(f"   Multi-GPU: {config.training.use_multi_gpu}\n")
    
    def _create_trainer(self) -> ImagenTrainer:
        """Create ImagenTrainer with optimal settings"""
        # Determine precision
        if self.config.training.use_amp:
            if self.config.training.precision == 'bf16':
                precision = 'bf16'
                fp16 = False
            else:  # fp16
                precision = 'fp16'
                fp16 = True
        else:
            precision = 'no'
            fp16 = False
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        trainer = ImagenTrainer(
            self.imagen,
            device=torch.device(self.config.training.device),
            
            # Mixed precision
            fp16=fp16,
            precision=precision if not fp16 else None,
            
            # Learning rate
            lr=self.config.training.learning_rate,
            eps=self.config.training.eps,
            beta1=self.config.training.betas[0],
            beta2=self.config.training.betas[1],
            
            # Gradient clipping
            max_grad_norm=self.config.training.max_grad_norm,
            
            # Warmup
            warmup_steps=self.config.training.warmup_steps,
            cosine_decay_max_steps=self.config.training.cosine_max_steps,
            
            # EMA (Exponential Moving Average)
            use_ema=True,
            ema_beta=0.9999,
            ema_update_after_step=100,
            ema_update_every=10,
            
            # Accelerate settings for multi-GPU
            accelerate_split_batches=True,
            accelerate_mixed_precision=precision,
            
            # Checkpointing (ensure dir exists before ImagenTrainer init)
            checkpoint_path=self.config.checkpoint_dir,
            checkpoint_every=self.config.training.save_every * len(self.train_loader),
            max_checkpoints_keep=self.config.training.keep_last_n_checkpoints,
            
            # Logging
            verbose=True
        )
        
        return trainer
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        loss_epoch = []
        
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Training Epoch {epoch}"
        )
        
        for iteration, batch in progress_bar:
            # Move to device
            batch = batch.to(self.config.training.device, non_blocking=True).float()
            
            # DICOM dataset yields (B, C, H, W); sequence data yields (B, T, C, H, W)
            # Normalise to 5D: (B, T, C, H, W)
            if batch.dim() == 4:
                # (B, C, H, W) → (B, 1, C, H, W)
                batch = batch.unsqueeze(1)
            
            bsize = batch.shape[0]
            ntime = batch.shape[1]
            H, W = batch.shape[-2], batch.shape[-1]
            
            # Split: left half = condition, right half = target
            batch_cond = batch[..., :W//2]
            batch = batch[..., W//2:]
            
            # Permute to [B, C, T, H, W]
            batch = batch.permute([0, 2, 1, 3, 4])
            batch_cond = batch_cond.permute([0, 2, 1, 3, 4])
            
            # Training step (trainer handles AMP, gradient accumulation, etc.)
            loss = self.trainer(
                batch,
                cond_images=batch_cond,
                unet_number=1,
                ignore_time=False
            )
            
            # Update
            self.trainer.update(unet_number=1)
            
            loss_epoch.append(loss)
            
            # Update progress bar
            if len(loss_epoch) > 0:
                avg_loss = sum(loss_epoch) / len(loss_epoch)
                progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        return sum(loss_epoch) / len(loss_epoch)
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate model.
        
        Returns:
            Average validation loss
        """
        self.trainer.eval()
        loss_valid = []
        
        for batch in tqdm(self.valid_loader, desc="Validating"):
            batch = batch.to(self.config.training.device, non_blocking=True).float()
            
            # Normalise to 5D: (B, T, C, H, W)
            if batch.dim() == 4:
                batch = batch.unsqueeze(1)
            
            bsize = batch.shape[0]
            ntime = batch.shape[1]
            H, W = batch.shape[-2], batch.shape[-1]
            
            # Split
            batch_cond = batch[..., :W//2]
            batch = batch[..., W//2:]
            
            # Permute
            batch = batch.permute([0, 2, 1, 3, 4])
            batch_cond = batch_cond.permute([0, 2, 1, 3, 4])
            
            # Evaluate
            loss = self.trainer(
                batch,
                cond_images=batch_cond,
                unet_number=1,
                ignore_time=False
            )
            
            loss_valid.append(loss)
        
        self.trainer.train()
        return sum(loss_valid) / len(loss_valid)
    
    def train(self):
        """
        Main training loop.
        """
        print("\n🚀 Starting Diffusion Model Training\n")
        
        # Resume if specified
        start_epoch = 0
        if self.config.training.resume_from:
            resume_path = self.config.training.resume_from
            if resume_path == 'latest':
                resume_path = os.path.join(self.config.checkpoint_dir, 'best_model_sofar')
            
            if os.path.exists(resume_path):
                print(f"📂 Resuming from {resume_path}")
                self.trainer.load(resume_path)
                
                # Try to get epoch from file
                epoch_file = resume_path + '_epoch'
                if os.path.exists(epoch_file):
                    epochs = np.loadtxt(epoch_file)
                    start_epoch = int(epochs[0]) + 1
            else:
                print(f"⚠️  Resume path not found: {resume_path}")
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(start_epoch, self.config.training.epoch_num):
            self.logger.on_epoch_start(epoch)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.training.eval_every == 0:
                valid_loss = self.validate()
            else:
                valid_loss = None
            
            # Save best model
            current_loss = valid_loss if valid_loss is not None else train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(self.config.checkpoint_dir, 'best_model_sofar')
                self.trainer.save(best_path)
                np.savetxt(
                    best_path + '_epoch',
                    np.ones(2) * epoch
                )
                print(f"⭐ New best model! Loss: {current_loss:.6f}")
            
            # Custom logging
            self.logger.metrics_tracker.update(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss if valid_loss is not None else train_loss,
                learning_rate=self.trainer.get_lr(1)
            )
            
            # Print summary
            print(f"\n📈 Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            if valid_loss is not None:
                print(f"  Valid Loss: {valid_loss:.6f}")
            print(f"  Best Loss: {best_loss:.6f}")
            print(f"  Learning Rate: {self.trainer.get_lr(1):.2e}")
            
            # Wandb logging
            if self.wandb_run is not None:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'best_loss': best_loss,
                    'learning_rate': self.trainer.get_lr(1),
                }
                if valid_loss is not None:
                    log_dict['valid_loss'] = valid_loss
                self.wandb_run.log(log_dict)
            
            # Save metrics every epoch
            self.logger.metrics_tracker.save()
            self.logger.metrics_tracker.plot()
        
        print("\n🎉 Training complete!\n")
        
        # Final save
        final_path = os.path.join(self.config.checkpoint_dir, 'final_model')
        self.trainer.save(final_path)
        print(f"💾 Final model saved to {final_path}")


def create_diffusion_model(config: ExperimentConfig) -> ElucidatedImagen:
    """
    Create diffusion model from configuration.
    
    Args:
        config: Experiment configuration
    
    Returns:
        ElucidatedImagen model
    """
    # Create UNet
    unet = Unet3D(
        dim=config.model.unet_dim,
        cond_images_channels=1,
        memory_efficient=config.model.memory_efficient,
        dim_mults=config.model.dim_mults
    ).to(torch.device(config.training.device))
    
    # Create Imagen
    imagen = ElucidatedImagen(
        unets=(unet,),
        image_sizes=(config.data.image_height,),
        image_width=(config.data.image_width,),
        channels=1,
        random_crop_sizes=None,
        
        # Sampling
        num_sample_steps=config.model.num_sample_steps,
        cond_drop_prob=config.model.cond_drop_prob,
        
        # Noise schedule
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        sigma_data=config.model.sigma_data,
        rho=config.model.rho,
        
        # Training distribution
        P_mean=config.model.P_mean,
        P_std=config.model.P_std,
        
        # Stochastic sampling
        S_churn=config.model.S_churn,
        S_tmin=config.model.S_tmin,
        S_tmax=config.model.S_tmax,
        S_noise=config.model.S_noise,
        
        # Other
        condition_on_text=False,
        auto_normalize_img=False
    ).to(torch.device(config.training.device))
    
    return imagen


if __name__ == '__main__':
    print("Diffusion Trainer Module")
    print("Use train_diffusion.py to run training")
