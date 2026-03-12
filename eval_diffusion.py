#!/usr/bin/env python3
"""
Evaluation Script for Diffusion Model
=====================================

Evaluates a trained diffusion model by generating samples and computing metrics.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import ExperimentConfig
from data.optimized_data import create_dataloaders_from_config
from train_test_spatial.optimized_trainer import create_diffusion_model
from mimagen_pytorch import Imagen
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save generated samples')
    parser.add_argument('--cond_scale', type=float, default=1.0,
                        help='Conditioning scale for sampling')
    parser.add_argument('--gen_batch_size', type=int, default=1,
                        help='Batch size during generation (reduce if OOM)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def load_model(checkpoint_path, config):
    """Load diffusion model from checkpoint."""
    
    # Create model architecture
    imagen = create_diffusion_model(config)
    
    checkpoint_file = Path(checkpoint_path)
    
    if checkpoint_file.is_dir():
        # Prefer best_model_sofar inside the directory, fall back to last .pt
        candidate = checkpoint_file / 'best_model_sofar'
        if candidate.exists():
            checkpoint_file = candidate
        else:
            pt_files = sorted(checkpoint_file.glob('checkpoint.*.pt'))
            if not pt_files:
                raise ValueError(f"No usable checkpoint found in {checkpoint_path}")
            checkpoint_file = pt_files[-1]  # latest step
    
    print(f"Loading checkpoint: {checkpoint_file}")
    loaded_obj = torch.load(checkpoint_file, map_location='cpu')
    
    # ImagenTrainer.save() stores model under 'model' key
    state_dict = loaded_obj['model'] if isinstance(loaded_obj, dict) and 'model' in loaded_obj else loaded_obj
    
    try:
        imagen.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting strict=False load")
        imagen.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Model loaded ({sum(p.numel() for p in imagen.parameters())/1e6:.2f}M params)")
    return imagen


def generate_samples(imagen, cond_images, device, cond_scale=1.0, gen_batch_size=1):
    """
    Generate samples conditioned on sinogram images.
    
    Args:
        cond_images: Sinogram tensors, shape (N, 1, T, H, W_half)
        Returns generated FBP reconstructions of same shape.
    """
    imagen.eval()
    imagen.to(device)
    cond_images = cond_images.to(device)
    
    print(f"\nGenerating {cond_images.shape[0]} samples (cond_scale={cond_scale}, gen_batch_size={gen_batch_size})...")
    
    all_samples = []
    batch_size = min(gen_batch_size, cond_images.shape[0])
    
    # T is the time/frame dimension in (B, C, T, H, W)
    video_frames = cond_images.shape[2]

    with torch.no_grad():
        for start in tqdm(range(0, cond_images.shape[0], batch_size), desc="Generating"):
            cond_batch = cond_images[start:start + batch_size]
            samples = imagen.sample(
                batch_size=cond_batch.shape[0],
                cond_images=cond_batch,
                cond_scale=cond_scale,
                video_frames=video_frames,
                return_all_unet_outputs=False
            )
            all_samples.append(samples.cpu())
    
    return torch.cat(all_samples, dim=0).numpy()


def compute_metrics(generated, targets):
    """Compute metrics between generated and target samples."""
    
    # Ensure same number of samples
    n = min(len(generated), len(targets))
    generated = generated[:n]
    targets = targets[:n]
    
    # MSE, MAE
    mse = np.mean((generated - targets) ** 2)
    mae = np.mean(np.abs(generated - targets))
    rmse = np.sqrt(mse)
    
    # PSNR
    mse_per_sample = np.mean((generated - targets) ** 2, axis=(1, 2, 3))
    max_val = np.max(targets)
    psnr = 10 * np.log10(max_val ** 2 / (mse_per_sample + 1e-10))
    avg_psnr = np.mean(psnr)

    # SSIM — computed per sample over (H, W), then averaged
    avg_ssim = None
    if SSIM_AVAILABLE:
        ssim_scores = []
        for i in range(n):
            # generated/targets shape: (C, H, W); take channel 0
            g = generated[i, 0].astype(np.float32)
            t = targets[i, 0].astype(np.float32)
            data_range = float(max(t.max() - t.min(), 1e-8))
            ssim_scores.append(ssim(g, t, data_range=data_range))
        avg_ssim = float(np.mean(ssim_scores))

    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'psnr': float(avg_psnr),
        'ssim': avg_ssim,
        'num_samples': n
    }
    
    return metrics


def visualize_samples(samples, targets, save_path):
    """Visualize generated vs target samples."""
    
    num_display = min(4, len(samples))
    fig, axes = plt.subplots(num_display, 3, figsize=(12, 3*num_display))
    
    if num_display == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_display):
        # Show generated sample
        if samples[i].shape[0] == 1:  # Grayscale
            axes[i, 0].imshow(samples[i, 0], cmap='viridis')
        else:  # RGB or multi-channel
            axes[i, 0].imshow(samples[i, 0], cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1}: Generated')
        axes[i, 0].axis('off')
        
        # Show target
        if targets[i].shape[0] == 1:
            axes[i, 1].imshow(targets[i, 0], cmap='viridis')
        else:
            axes[i, 1].imshow(targets[i, 0], cmap='viridis')
        axes[i, 1].set_title(f'Sample {i+1}: Target')
        axes[i, 1].axis('off')
        
        # Show difference
        diff = np.abs(samples[i, 0] - targets[i, 0])
        axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Sample {i+1}: |Difference|')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    # Check device
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load config
    if args.config is None:
        ckpt = Path(args.checkpoint)
        # Search upward from checkpoint location to find config.json
        search_dirs = [ckpt.parent, ckpt.parent.parent, ckpt.parent.parent.parent]
        candidates = []
        for d in search_dirs:
            candidates += [d / 'logs' / 'config.json', d / 'config.json']
        candidates.append(ckpt / 'config.json')
        for candidate in candidates:
            if candidate.exists():
                args.config = str(candidate)
                break
        if args.config is None:
            raise ValueError("config.json not found. Pass --config explicitly.")
    
    print(f"Loading config from {args.config}")
    config = ExperimentConfig.load(args.config)
    config.training.device = args.device
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    imagen = load_model(args.checkpoint, config)
    imagen.to(args.device)
    
    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    # Use dataset_type from config (already has correct splits saved)
    mode = config.data.dataset_type if hasattr(config.data, 'dataset_type') else 'diffusion'
    dataloaders = create_dataloaders_from_config(config, mode=mode)
    dataloader = dataloaders['valid'] if args.split == 'valid' else dataloaders['train']
    
    # Collect batches — apply same sparse-angle masking as training
    sparse_strategy = getattr(config.data, 'sparse_strategy', 'angle_25pct')
    cond_list, target_list = [], []
    for batch in dataloader:
        batch = batch.float()
        if batch.dim() == 4:          # (B, C, H, W) → (B, 1, C, H, W)
            batch = batch.unsqueeze(1)
        # Pad H to multiple of 16 (mirrors training _pad_h16)
        B_, T_, C_, H_, W_ = batch.shape
        pad_h = (16 - H_ % 16) % 16
        if pad_h > 0:
            import torch.nn.functional as _F
            batch = batch.reshape(B_ * T_, C_, H_, W_)
            batch = _F.pad(batch, (0, 0, 0, pad_h), mode='reflect')
            batch = batch.reshape(B_, T_, C_, H_ + pad_h, W_)
        # Build angle-masked condition (same as _apply_sparse_split in trainer)
        H_pad = batch.shape[-2]
        batch_target = batch
        batch_cond = torch.zeros_like(batch_target)
        if sparse_strategy == 'angle_25pct':
            cond_indices = list(range(0, H_, 4))
        elif sparse_strategy == 'angle_12pct':
            cond_indices = list(range(0, H_, 8))
        elif sparse_strategy == 'angle_block_50pct':
            cond_indices = list(range(0, H_ // 2))
        elif sparse_strategy == 'angle_step_50pct':
            cond_indices = list(range(0, H_, 2))
        elif sparse_strategy == 'angle_limited_25pct':
            cond_indices = list(range(0, int(H_ * 0.75), 4))
        else:
            cond_indices = list(range(0, H_, 4))
        batch_cond[..., cond_indices, :] = batch_target[..., cond_indices, :]
        # Permute to [B, C, T, H, W]
        cond   = batch_cond.permute(0, 2, 1, 3, 4)
        target = batch_target.permute(0, 2, 1, 3, 4)
        cond_list.append(cond)
        target_list.append(target)
        if sum(c.shape[0] for c in cond_list) >= args.num_samples:
            break
    
    cond_all   = torch.cat(cond_list,   dim=0)[:args.num_samples]
    target_all = torch.cat(target_list, dim=0)[:args.num_samples]
    print(f"Collected {cond_all.shape[0]} samples — cond {tuple(cond_all.shape)}, target {tuple(target_all.shape)}")
    
    # Generate
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)
    try:
        generated = generate_samples(imagen, cond_all, args.device, args.cond_scale, args.gen_batch_size)
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        raise
    
    # Squeeze time dim for metrics: (B, C, T, H, W) → (B, C, H, W)
    generated_np = generated[:, :, 0] if generated.ndim == 5 else generated
    targets_np   = target_all.numpy()[:, :, 0] if target_all.ndim == 5 else target_all.numpy()
    
    # Metrics
    print("\n" + "="*70)
    print("METRICS")
    print("="*70)
    metrics = compute_metrics(generated_np, targets_np)
    print(f"Samples : {metrics['num_samples']}")
    print(f"MSE     : {metrics['mse']:.6f}")
    print(f"MAE     : {metrics['mae']:.6f}")
    print(f"RMSE    : {metrics['rmse']:.6f}")
    print(f"PSNR    : {metrics['psnr']:.2f} dB")
    if metrics['ssim'] is not None:
        print(f"SSIM    : {metrics['ssim']:.4f}")
    else:
        print("SSIM    : N/A (install scikit-image)")
    print("="*70 + "\n")
    
    # Save results
    if args.save_samples:
        output_dir = Path(args.checkpoint).parent / 'eval_results'
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / f'generated_{args.split}.npy',  generated_np)
        np.save(output_dir / f'targets_{args.split}.npy',    targets_np)
        np.save(output_dir / f'cond_{args.split}.npy',       cond_all.numpy()[:, :, 0])
        
        with open(output_dir / f'metrics_{args.split}.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        viz_path = output_dir / f'samples_viz_{args.split}.png'
        visualize_samples(generated_np, targets_np, viz_path)
        
        print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
