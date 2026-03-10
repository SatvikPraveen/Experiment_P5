#!/usr/bin/env python3
"""
Main Training Script for Sequential Model
==========================================

Usage:
    # Use predefined configuration
    python train_sequential.py --preset optimized_default
    
    # Custom configuration
    python train_sequential.py --config myconfig.json
    
    # Override specific parameters
    python train_sequential.py --preset optimized_default --batch_size 32 --epochs 200
"""

import argparse
import sys
import os
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import get_preset_config, ExperimentConfig
from data.optimized_data import create_dataloaders_from_config
from transformer.sequentialModel import SequentialModel
from train_test_seq.optimized_trainer import OptimizedSequentialTrainer
from util.optimized_utils import set_random_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available - install with: pip install wandb")


def _validate_config(config):
    """Validate configuration before training."""
    # Check data paths exist
    for path in config.data.data_location:
        if not os.path.exists(path):
            print(f"\n⚠️  WARNING: Data file not found: {path}")
            print("   Please update data_location in configs/base_config.py")
            print("   or use a custom config file.")
            print("   Training will continue but may fail during data loading.\n")
            break
    
    # Check coarse dimension matches n_embd
    expected_embd = config.model.coarse_dim[0] * config.model.coarse_dim[1] * 2
    if expected_embd != config.model.n_embd:
        print(f"\n❌ ERROR: Configuration mismatch!")
        print(f"   coarse_dim {config.model.coarse_dim} * 2 = {expected_embd}")
        print(f"   but n_embd = {config.model.n_embd}")
        print(f"   These must match!\n")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Sequential Transformer Model')
    
    # Configuration
    parser.add_argument('--preset', type=str, default='optimized_default',
                        help='Preset configuration name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')
    
    # Override options
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda:0, cuda:1, etc.)')
    parser.add_argument('--num_workers', type=int, help='DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable mixed precision')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity (username/team)')
    
    # Multi-GPU
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
    
    # Debugging
    parser.add_argument('--debug', action='store_true', help='Debug mode (single epoch)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        print(f"Using preset: {args.preset}")
        config = get_preset_config(args.preset)
        config.mode = 'sequential'
    
    # Apply overrides
    if args.name:
        config.name = args.name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epoch_num = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    if args.num_workers:
        config.data.num_workers = args.num_workers
    if args.amp is not None:
        config.training.use_amp = args.amp
    if args.resume:
        config.training.resume_from = args.resume
    if args.multi_gpu:
        config.training.use_multi_gpu = True
    if args.seed is not None:
        config.training.random_seed = args.seed
    if args.wandb is not None:
        config.training.use_wandb = args.wandb
    if args.wandb_project:
        config.training.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.training.wandb_entity = args.wandb_entity
    
    # Set random seed for reproducibility
    set_random_seed(config.training.random_seed, config.training.deterministic)
    
    # Debug mode
    if args.debug:
        config.training.epoch_num = 2
        config.training.save_every = 1
        config.training.eval_every = 1
        config.data.train_span = 200
        config.data.valid_span = 100
        config.data.seq_length_valid = 50  # Reduce for debug mode
        config.name = f"{config.name}_debug"
        print("\n⚠️  DEBUG MODE: 2 epochs only\n")
    
    # Validate configuration
    _validate_config(config)
    
    # Print configuration
    config.print_summary()
    
    # Save configuration
    config.save()
    
    # Initialize Wandb
    wandb_run = None
    if config.training.use_wandb and WANDB_AVAILABLE:
        # Create unique project name: experimentp5-seq-<preset>-seed<seed>
        wandb_project = f"{config.training.wandb_project}-seq-{config.name}"
        
        wandb_run = wandb.init(
            project=wandb_project,
            entity=config.training.wandb_entity,
            name=f"seed{config.training.random_seed}",
            config={
                'preset': args.preset,
                'mode': 'sequential',
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'epochs': config.training.epoch_num,
                'seed': config.training.random_seed,
                'image_size': f"{config.data.image_height}x{config.data.image_width}",
                'n_layers': config.model.n_layer,
                'n_embd': config.model.n_embd,
                'unet_dim': config.model.unet_dim,
            },
            tags=['sequential', args.preset, f'seed{config.training.random_seed}'],
        )
        print(f"\n📊 Wandb initialized: {wandb_project}")
        print(f"   Run: {wandb_run.name}")
        print(f"   URL: {wandb_run.url}\n")
    elif config.training.use_wandb and not WANDB_AVAILABLE:
        print("\n⚠️  Wandb requested but not available. Install with: pip install wandb\n")
    
    # Validate coarse dimension
    assert (config.model.coarse_dim[0] * config.model.coarse_dim[1] * 2 == config.model.n_embd), \
        f"coarse_dim ({config.model.coarse_dim}) * 2 must equal n_embd ({config.model.n_embd})"
    
    # Create dataloaders
    print("\n📊 Creating dataloaders...")
    dataloaders = create_dataloaders_from_config(config, mode='sequential')
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Valid batches: {len(valid_loader)}")
    
    # Create model
    print("\n🏗️  Creating sequential transformer model...")
    
    # Create config object for SequentialModel (it expects this format)
    class ModelConfig:
        def __init__(self, cfg):
            self.n_layer = cfg.model.n_layer
            self.n_ctx = cfg.model.n_ctx
            self.n_embd = cfg.model.n_embd
            self.n_head = cfg.model.n_head
            self.embd_pdrop = cfg.model.embd_pdrop
            self.attn_pdrop = cfg.model.attn_pdrop
            self.resid_pdrop = cfg.model.resid_pdrop
            self.activation_function = cfg.model.activation_function
            self.layer_norm_epsilon = cfg.model.layer_norm_epsilon
            self.initializer_range = cfg.model.initializer_range
            self.output_hidden_states = True
            self.output_attentions = True
    
    model_config = ModelConfig(config)
    model = SequentialModel(model_config).float()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Optional: torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("🔧 Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    
    # Create trainer
    print("\n⚙️  Initializing trainer...")
    trainer = OptimizedSequentialTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        wandb_run=wandb_run
    )
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        checkpoint_path = os.path.join(config.checkpoint_dir, 'interrupted.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'config': config
        }, checkpoint_path)
        print(f"💾 Checkpoint saved to {checkpoint_path}")
    
    # Finish wandb
    if wandb_run is not None:
        wandb_run.finish()
    
    print("\n✅ Training script complete!")
    print(f"\n📁 Model saved to: {config.checkpoint_dir}")
    print(f"   - Best model: {config.checkpoint_dir}/best_model.pt")
    print(f"   - Final model: {config.checkpoint_dir}/final_model.pt")
    print(f"   - Checkpoints: {config.checkpoint_dir}/checkpoint_epoch_*.pt\n")


if __name__ == '__main__':
    main()
