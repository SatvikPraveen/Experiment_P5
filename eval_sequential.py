#!/usr/bin/env python3
"""
Evaluation Script for Sequential Model
======================================

Evaluates a trained sequential (transformer) model on validation/test data.
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import ExperimentConfig
from data.optimized_data import create_dataloaders_from_config
from transformer.sequentialModel import GPT2Transformer
from util.optimized_utils import TrainingLogger
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Sequential Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config (auto-detected from checkpoint if not provided)')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def load_model_and_config(checkpoint_path, config_path=None):
    """Load model and configuration."""
    
    # Try to find config automatically
    if config_path is None:
        checkpoint_dir = Path(checkpoint_path).parent.parent
        potential_config = checkpoint_dir / 'config.json'
        if potential_config.exists():
            config_path = str(potential_config)
            print(f"Auto-detected config: {config_path}")
    
    if config_path is None:
        raise ValueError("Config path not provided and could not be auto-detected")
    
    # Load configuration
    config = ExperimentConfig.load(config_path)
    
    # Create model
    model = GPT2Transformer(
        time_length=config.model.time_steps,
        num_dets=config.data.num_detectors,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model (no metadata)")
    
    return model, config


def evaluate(model, dataloader, device, num_samples=None):
    """Evaluate model on dataset."""
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if num_samples and total_samples >= num_samples:
                break
            
            # Move to device
            if isinstance(batch, dict):
                x = batch['input'].to(device)
                y = batch['target'].to(device)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
            
            # Forward pass
            predictions = model(x)
            
            # Compute loss
            loss = criterion(predictions, y)
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Save predictions
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluated {total_samples} samples, current loss: {loss.item():.6f}")
    
    avg_loss = total_loss / total_samples
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute additional metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    # Relative error
    relative_error = np.mean(np.abs(all_predictions - all_targets) / (np.abs(all_targets) + 1e-8))
    
    metrics = {
        'loss': avg_loss,
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'relative_error': float(relative_error),
        'num_samples': total_samples
    }
    
    return metrics, all_predictions, all_targets


def visualize_predictions(predictions, targets, save_path):
    """Create visualization of predictions vs targets."""
    
    num_samples = min(4, len(predictions))
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Plot target
        axes[i, 0].plot(targets[i].flatten())
        axes[i, 0].set_title(f'Sample {i+1}: Target')
        axes[i, 0].set_ylabel('Value')
        
        # Plot prediction
        axes[i, 1].plot(predictions[i].flatten(), label='Prediction')
        axes[i, 1].plot(targets[i].flatten(), alpha=0.5, label='Target')
        axes[i, 1].set_title(f'Sample {i+1}: Prediction vs Target')
        axes[i, 1].legend()
        axes[i, 1].set_ylabel('Value')
    
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
    
    # Load model and config
    model, config = load_model_and_config(args.checkpoint, args.config)
    
    # Override batch size
    config.training.batch_size = args.batch_size
    
    # Create dataloader
    print(f"\nLoading {args.split} dataset...")
    dataloaders = create_dataloaders_from_config(config, mode='sequential')
    
    if args.split == 'valid':
        dataloader = dataloaders['valid']
    elif args.split == 'train':
        dataloader = dataloaders['train']
    else:
        raise ValueError(f"Split '{args.split}' not available. Use 'train' or 'valid'")
    
    print(f"Dataset size: {len(dataloader.dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70 + "\n")
    
    metrics, predictions, targets = evaluate(
        model, dataloader, args.device, args.num_samples
    )
    
    # Print metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Samples evaluated: {metrics['num_samples']}")
    print(f"MSE:              {metrics['mse']:.6f}")
    print(f"MAE:              {metrics['mae']:.6f}")
    print(f"RMSE:             {metrics['rmse']:.6f}")
    print(f"Relative Error:   {metrics['relative_error']:.4%}")
    print("="*70 + "\n")
    
    # Save predictions
    if args.save_predictions:
        output_dir = Path(args.checkpoint).parent.parent / 'eval_results'
        output_dir.mkdir(exist_ok=True)
        
        pred_path = output_dir / f'predictions_{args.split}.npy'
        target_path = output_dir / f'targets_{args.split}.npy'
        metrics_path = output_dir / f'metrics_{args.split}.txt'
        
        np.save(pred_path, predictions)
        np.save(target_path, targets)
        
        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Predictions saved to {pred_path}")
        print(f"Targets saved to {target_path}")
        print(f"Metrics saved to {metrics_path}")
        
        # Create visualization
        viz_path = output_dir / f'predictions_viz_{args.split}.png'
        visualize_predictions(predictions, targets, viz_path)


if __name__ == '__main__':
    main()
