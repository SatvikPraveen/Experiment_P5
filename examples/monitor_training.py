#!/usr/bin/env python3
"""
Example: Training Monitoring and Analysis
=========================================

Shows how to monitor training progress and analyze results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metrics(experiment_name):
    """Load training metrics from JSON."""
    metrics_file = Path(f'../output/{experiment_name}/metrics/metrics.json')
    with open(metrics_file, 'r') as f:
        return json.load(f)

def plot_training_curves(experiment_name, save_path=None):
    """Plot training and validation loss curves."""
    metrics = load_metrics(experiment_name)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    if 'train_loss' in metrics:
        axes[0].plot(metrics['train_loss'], label='Training Loss', linewidth=2)
    if 'valid_loss' in metrics:
        axes[0].plot(metrics['valid_loss'], label='Validation Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{experiment_name}: Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in metrics:
        axes[1].plot(metrics['learning_rate'], linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_training(experiment_name):
    """Print training analysis."""
    metrics = load_metrics(experiment_name)
    
    print("="*70)
    print(f"Training Analysis: {experiment_name}")
    print("="*70)
    
    if 'train_loss' in metrics:
        train_losses = metrics['train_loss']
        print(f"Training Loss:")
        print(f"  Initial:  {train_losses[0]:.6f}")
        print(f"  Final:    {train_losses[-1]:.6f}")
        print(f"  Best:     {min(train_losses):.6f}")
        print(f"  Reduction: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
        print()
    
    if 'valid_loss' in metrics:
        valid_losses = metrics['valid_loss']
        print(f"Validation Loss:")
        print(f"  Initial:  {valid_losses[0]:.6f}")
        print(f"  Final:    {valid_losses[-1]:.6f}")
        print(f"  Best:     {min(valid_losses):.6f}")
        print(f"  Epoch of best: {np.argmin(valid_losses) + 1}")
        
        # Check for overfitting
        last_10 = valid_losses[-10:]
        if len(last_10) > 0:
            trend = np.mean(np.diff(last_10))
            if trend > 0:
                print(f"  ⚠️  Validation loss increasing (possible overfitting)")
            else:
                print(f"  ✅  Validation loss still decreasing")
        print()
    
    # Training time
    if 'epoch_time' in metrics:
        epoch_times = metrics['epoch_time']
        print(f"Training Time:")
        print(f"  Avg epoch: {np.mean(epoch_times):.1f}s")
        print(f"  Total:     {sum(epoch_times)/3600:.2f} hours")
        print()
    
    # GPU stats
    if 'gpu_memory_used' in metrics:
        gpu_mem = metrics['gpu_memory_used']
        print(f"GPU Memory:")
        print(f"  Avg:  {np.mean(gpu_mem):.1f} MB")
        print(f"  Peak: {max(gpu_mem):.1f} MB")
        print()
    
    print("="*70)

def compare_experiments(exp_names):
    """Compare multiple experiments."""
    print("="*70)
    print("Experiment Comparison")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name in exp_names:
        try:
            metrics = load_metrics(name)
            if 'valid_loss' in metrics:
                ax.plot(metrics['valid_loss'], label=name, linewidth=2)
        except Exception as e:
            print(f"Could not load {name}: {e}")
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Experiment Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150)
    print("Comparison plot saved to experiment_comparison.png")
    plt.close()
    
    # Print summary table
    print(f"\n{'Experiment':<30} {'Best Val Loss':<15} {'Final Val Loss':<15}")
    print("-"*70)
    for name in exp_names:
        try:
            metrics = load_metrics(name)
            if 'valid_loss' in metrics:
                best = min(metrics['valid_loss'])
                final = metrics['valid_loss'][-1]
                print(f"{name:<30} {best:<15.6f} {final:<15.6f}")
        except:
            pass
    print("="*70)

# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == '__main__':
    # Example 1: Analyze single experiment
    print("Example 1: Single Experiment Analysis\n")
    experiment = 'seq_model'  # Replace with your experiment name
    
    try:
        analyze_training(experiment)
        plot_training_curves(experiment, f'{experiment}_curves.png')
    except FileNotFoundError:
        print(f"Experiment '{experiment}' not found.")
        print("Train a model first!")
    
    # Example 2: Compare multiple experiments
    print("\n\nExample 2: Compare Experiments\n")
    experiments = [
        'seq_model',
        'seq_model_large',
        'seq_model_fast'
    ]
    
    try:
        compare_experiments(experiments)
    except Exception as e:
        print(f"Comparison failed: {e}")
        print("Make sure experiments exist in output/")
    
    print("\n✅ Analysis complete!")
