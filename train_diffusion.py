#!/usr/bin/env python3
"""
Main Training Script for Diffusion Model
=========================================

Usage:
    # Use predefined configuration
    python train_diffusion.py --preset 720_816_medium_unet
    
    # Custom configuration
    python train_diffusion.py --config myconfig.json
    
    # With sequential model    python train_diffusion.py --preset optimized_default --seq_model_path ../output/seq_model/best.pt
"""

import argparse
import sys
import os
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.base_config import get_preset_config, ExperimentConfig
from data.optimized_data import create_dataloaders_from_config
from train_test_spatial.optimized_trainer import OptimizedDiffusionTrainer, create_diffusion_model
from util.optimized_utils import set_random_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available - install with: pip install wandb")


def _validate_config(config):
    """Validate configuration before training."""
    # Check data paths exist (skip for diffusion if using DICOM)
    if hasattr(config.data, 'data_location') and config.data.data_location:
        for path in config.data.data_location:
            if not os.path.exists(path):
                print(f"\n⚠️  WARNING: Data file not found: {path}")
                print("   Please update data paths in configuration.\n")
                break


def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Model')
    
    # Configuration
    parser.add_argument('--preset', type=str, default='optimized_default',
                        help='Preset configuration name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')
    
    # Sequential model (optional)
    parser.add_argument('--seq_model_path', type=str, default=None,
                        help='Path to trained sequential model (optional)')
    parser.add_argument('--seq_config', type=str, default=None,
                        help='Sequential model config (optional)')
    
    # Override options
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--batch_size_valid', type=int, help='Validation batch size (defaults to training batch size)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda:0, cuda:1, etc.)')
    parser.add_argument('--num_workers', type=int, help='DataLoader workers')
    parser.add_argument('--unet_dim', type=int, help='UNet base dimension')
    parser.add_argument('--amp', action='store_true', default=None, help='Enable mixed precision')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Disable mixed precision')
    parser.set_defaults(amp=None)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Save a checkpoint every N epochs (e.g. 10 → saves at epoch 10, 20, 30, ...)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, help='Output directory for checkpoints and logs')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='Disable wandb logging')
    parser.set_defaults(wandb=None)
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity (username/team)')
    
    # Multi-GPU
    parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
    
    # DICOM + FBP mode
    parser.add_argument('--dicom', action='store_true', help='Use DICOM+FBP pipeline instead of .npy data')
    parser.add_argument('--dicom_path', type=str, default=None, help='Path to DICOM series directory (default: data/Dataset)')
    
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
        config.mode = 'diffusion'
    
    # Apply overrides
    if args.name:
        config.name = args.name
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.batch_size_valid:
        config.training.batch_size_valid = args.batch_size_valid
    if args.epochs:
        config.training.epoch_num = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    if args.num_workers:
        config.data.num_workers = args.num_workers
    if args.unet_dim:
        config.model.unet_dim = args.unet_dim
    if args.amp is not None:
        config.training.use_amp = args.amp
    if args.resume:
        config.training.resume_from = args.resume
    if args.multi_gpu:
        config.training.use_multi_gpu = True
    if args.seed is not None:
        config.training.random_seed = args.seed
    if args.output_dir:
        config.set_output_dir(args.output_dir)
    if args.wandb is not None:
        config.training.use_wandb = args.wandb
    if args.wandb_project:
        config.training.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.training.wandb_entity = args.wandb_entity
    
    # DICOM mode: use raw sinograms (dicom_sino) for the sparse-view CT
    # projection-angle reconstruction task.  dicom_fbp (FBP pixel images)
    # is a separate image-domain pipeline and must NOT be used here.
    if args.dicom:
        config.data.dataset_type = 'dicom_sino'
    if args.dicom_path:
        config.data.dicom_path = args.dicom_path
    if config.data.dataset_type in ('dicom_fbp', 'dicom_sino'):
        from pathlib import Path as _Path
        # If a pre-built sino_cache exists, derive splits from the number of
        # consecutive series-0 files (sino_s0_i<N>_...) — which is exactly
        # what DICOMDataset iterates in cache_only mode.  Counting all *.npy
        # files is wrong because the cache may hold multiple series.
        _sino_cache = getattr(config.data, 'sino_cache_dir', 'data/sino_cache')
        _npy_count = 0
        if _sino_cache and _Path(_sino_cache).is_dir():
            _npy_count = sum(1 for f in os.listdir(_sino_cache) if f.endswith('.npy'))
        if _npy_count > 0:
            _TOTAL_SLICES = _npy_count
            print(f"\n🏥 DICOM mode (sino_cache): {_TOTAL_SLICES} total sinograms in {_sino_cache}")
        else:
            # Fall back to counting .dcm files when no cache exists
            _TOTAL_SLICES = len(list(_Path(config.data.dicom_path).glob('*.dcm')))
            print(f"\n🏥 DICOM mode: {_TOTAL_SLICES} slices from {config.data.dicom_path}")
        config.data.train_start = 0
        config.data.train_span = int(_TOTAL_SLICES * 0.75)
        config.data.valid_start = config.data.train_span
        config.data.valid_span = int(_TOTAL_SLICES * 0.15)
        config.data.test_start = config.data.valid_start + config.data.valid_span
        config.data.test_span = _TOTAL_SLICES - config.data.test_start
        print(f"   Train: {config.data.train_start}→{config.data.train_start + config.data.train_span}")
        print(f"   Valid: {config.data.valid_start}→{config.data.valid_start + config.data.valid_span}")
        print(f"   Test:  {config.data.test_start}→{config.data.test_start + config.data.test_span}\n")
    
    # Set random seed for reproducibility
    set_random_seed(config.training.random_seed, config.training.deterministic)
    
    # Debug mode
    if args.save_every is not None:
        config.training.save_every = args.save_every

    if args.debug:
        config.training.epoch_num = 2
        config.training.save_every = 1
        config.training.eval_every = 1
        config.data.train_span = 50
        config.data.valid_span = 20
        config.data.seq_length_valid = 10  # Reduce for debug mode
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
        # Create unique project name: experimentp5-diff-<preset>-seed<seed>
        wandb_project = f"{config.training.wandb_project}-diff-{config.name}"
        
        wandb_run = wandb.init(
            project=wandb_project,
            entity=config.training.wandb_entity,
            name=f"seed{config.training.random_seed}",
            config={
                'preset': args.preset,
                'mode': 'diffusion',
                'batch_size': config.training.batch_size,
                'learning_rate': config.training.learning_rate,
                'epochs': config.training.epoch_num,
                'seed': config.training.random_seed,
                'image_size': f"{config.data.image_height}x{config.data.image_width}",
                'unet_dim': config.model.unet_dim,
                'num_sample_steps': config.model.num_sample_steps,
            },
            tags=['diffusion', args.preset, f'seed{config.training.random_seed}'],
        )
        print(f"\n📊 Wandb initialized: {wandb_project}")
        print(f"   Run: {wandb_run.name}")
        print(f"   URL: {wandb_run.url}\n")
    elif config.training.use_wandb and not WANDB_AVAILABLE:
        print("\n⚠️  Wandb requested but not available. Install with: pip install wandb\n")
    
    # Load sequential model config if provided
    seq_config = None
    if args.seq_config:
        print(f"\n📂 Loading sequential model config from {args.seq_config}")
        seq_config = ExperimentConfig.load(args.seq_config)
    
    # Create dataloaders
    print("\n📊 Creating dataloaders...")
    dataloaders = create_dataloaders_from_config(config, mode=config.data.dataset_type)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Valid batches: {len(valid_loader)}")
    
    # Create diffusion model
    print("\n🏗️  Creating diffusion model...")
    imagen = create_diffusion_model(config)
    
    # Count parameters
    total_params = 0
    for unet in imagen.unets:
        if unet is not None:
            num_params = sum(p.numel() for p in unet.parameters())
            total_params += num_params
            print(f"   UNet parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Create trainer
    print("\n⚙️  Initializing trainer...")
    trainer = OptimizedDiffusionTrainer(
        config=config,
        imagen=imagen,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seq_model_config=seq_config,
        wandb_run=wandb_run
    )
    
    # Load sequential model if provided
    if args.seq_model_path:
        print(f"\n📂 Loading sequential model from {args.seq_model_path}")
        # This would be used for conditioning in future implementations
        # For now, just acknowledge it
        print("   ℹ️  Sequential model loading not yet implemented in trainer")
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Saving checkpoint...")
        checkpoint_path = os.path.join(config.checkpoint_dir, 'interrupted')
        trainer.trainer.save(checkpoint_path)
        print(f"💾 Checkpoint saved to {checkpoint_path}")
    
    # Finish wandb
    if wandb_run is not None:
        wandb_run.finish()
    
    print("\n✅ Training script complete!")
    print(f"\n📁 Model saved to: {config.checkpoint_dir}")
    print(f"   - Best model: {config.checkpoint_dir}/best_model_sofar")
    print(f"   - Final model: {config.checkpoint_dir}/final_model")
    print(f"   - Use trainer.load() to load these models\n")


if __name__ == '__main__':
    main()
