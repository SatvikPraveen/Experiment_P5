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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def load_model(checkpoint_path, config):
    """Load diffusion model from checkpoint."""
    
    # Create model
    imagen = create_diffusion_model(config)
    
    # Try to load checkpoint
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        # Try to find model file
        potential_files = list(checkpoint_file.glob('*.pt'))
        if not potential_files:
            raise ValueError(f"No .pt files found in {checkpoint_path}")
        checkpoint_file = potential_files[0]
        print(f"Using checkpoint: {checkpoint_file}")
    
    print(f"Loading model from {checkpoint_file}")
    
    # Load state dict
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load into model (may need to adjust for wrapped models)
    try:
        imagen.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Attempting to load with strict=False")
        imagen.load_state_dict(state_dict, strict=False)
    
    return imagen


def generate_samples(imagen, num_samples, device, cond_scale=1.0):
    """Generate samples from diffusion model."""
    
    imagen.eval()
    imagen.to(device)
    
    print(f"\nGenerating {num_samples} samples...")
    
    samples = []
    
    with torch.no_grad():
        # Generate in batches
        batch_size = min(4, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Sample from model
            # Note: This requires appropriate text/condition input
            # For unconditional generation:
            batch_samples = imagen.sample(
                batch_size=current_batch_size,
                cond_scale=cond_scale,
                return_all_unet_outputs=False
            )
            
            samples.append(batch_samples.cpu().numpy())
    
    samples = np.concatenate(samples, axis=0)
    return samples


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
    
    # SSIM would require skimage
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'psnr': float(avg_psnr),
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
        config_path = Path(args.checkpoint).parent / 'config.json'
        if not config_path.exists():
            config_path = Path(args.checkpoint) / 'config.json'
        if not config_path.exists():
            raise ValueError("Config not found, please specify --config")
        args.config = str(config_path)
    
    print(f"Loading config from {args.config}")
    config = ExperimentConfig.load(args.config)
    
    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    imagen = load_model(args.checkpoint, config)
    
    # Count parameters
    total_params = sum(p.numel() for p in imagen.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Load data for comparison
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    dataloaders = create_dataloaders_from_config(config, mode='diffusion')
    
    if args.split == 'valid':
        dataloader = dataloaders['valid']
    else:
        dataloader = dataloaders['train']
    
    # Get some target samples
    targets = []
    for batch in dataloader:
        if isinstance(batch, dict):
            targets.append(batch['target'].numpy())
        else:
            _, y = batch
            targets.append(y.numpy())
        
        if len(targets) * config.training.batch_size >= args.num_samples:
            break
    
    targets = np.concatenate(targets, axis=0)[:args.num_samples]
    print(f"Loaded {len(targets)} target samples")
    
    # Generate samples
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70)
    
    try:
        generated = generate_samples(
            imagen, args.num_samples, args.device, args.cond_scale
        )
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Note: Unconditional generation may require text embeddings.")
        print("This is a placeholder - adjust based on your model's conditioning.")
        return
    
    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70)
    
    metrics = compute_metrics(generated, targets)
    
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"MSE:   {metrics['mse']:.6f}")
    print(f"MAE:   {metrics['mae']:.6f}")
    print(f"RMSE:  {metrics['rmse']:.6f}")
    print(f"PSNR:  {metrics['psnr']:.2f} dB")
    print("="*70 + "\n")
    
    # Save results
    if args.save_samples:
        output_dir = Path(args.checkpoint).parent / 'eval_results'
        output_dir.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(output_dir / f'generated_{args.split}.npy', generated)
        np.save(output_dir / f'targets_{args.split}.npy', targets)
        
        # Save metrics
        with open(output_dir / f'metrics_{args.split}.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        # Visualize
        viz_path = output_dir / f'samples_viz_{args.split}.png'
        visualize_samples(generated, targets, viz_path)
        
        print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
