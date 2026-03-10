#!/usr/bin/env python3
"""
Example: Custom Training Configuration
======================================

Shows how to create and use custom configurations for experiments.
"""

from configs.base_config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

# ============================================
# EXAMPLE 1: Modify existing preset
# ============================================

config = ExperimentConfig.from_preset('optimized_default')

# Modify parameters
config.name = 'my_custom_experiment'
config.training.batch_size = 32
config.training.learning_rate = 5e-5
config.training.epoch_num = 150

# Save configuration
config.save()
print(f"Configuration saved to: {config.save_path}")

# Use for training:
# python train_sequential.py --config output/my_custom_experiment/config.json

# ============================================
# EXAMPLE 2: Create completely custom config
# ============================================

custom_config = ExperimentConfig(
    name='high_res_experiment',
    mode='sequential',
    
    # Data configuration
    data=DataConfig(
        data_type='bfs',
        data_dir='/path/to/my/data',
        num_detectors=1024,
        train_span=1000,
        valid_span=200,
        num_workers=16  # Lots of CPU cores
    ),
    
    # Model configuration
    model=ModelConfig(
        time_steps=50,
        hidden_dim=512,  # Larger model
        num_layers=12,
        num_heads=8,
        dropout=0.2,
        unet_dim=128,  # For diffusion
        unet_dim_mults=(1, 2, 4, 8)
    ),
    
    # Training configuration
    training=TrainingConfig(
        batch_size=16,
        epoch_num=200,
        learning_rate=1e-4,
        device='cuda',
        use_amp=True,
        use_multi_gpu=True,
        gradient_accumulation_steps=2,  # Effective batch size = 32
        save_every=10,
        eval_every=5
    )
)

custom_config.save()
print(f"Custom config saved to: {custom_config.save_path}")

# ============================================
# EXAMPLE 3: Dataset-specific configurations
# ============================================

# For DICOM/CT data
dicom_config = ExperimentConfig.from_preset('optimized_default')
dicom_config.name = 'ct_reconstruction'
dicom_config.mode = 'diffusion'
dicom_config.data.data_type = 'dicom'
dicom_config.data.num_detectors = 1024
dicom_config.data.angles_num = 180
dicom_config.data.image_size = (512, 512)
dicom_config.save()

# For BFS fluid dynamics data
bfs_config = ExperimentConfig.from_preset('720_816_medium_unet')
bfs_config.name = 'fluid_dynamics'
bfs_config.data.data_type = 'bfs'
bfs_config.data.train_span = 2000
bfs_config.data.valid_span = 500
bfs_config.save()

# ============================================
# EXAMPLE 4: Quick prototyping config
# ============================================

prototype_config = ExperimentConfig.from_preset('small_fast')
prototype_config.name = 'quick_test'
prototype_config.training.epoch_num = 10
prototype_config.data.train_span = 100
prototype_config.data.valid_span = 20
prototype_config.training.batch_size = 8
prototype_config.save()

print("\n✅ All configurations created!")
print("\nTo use them:")
print("  python train_sequential.py --config output/my_custom_experiment/config.json")
print("  python train_diffusion.py --config output/ct_reconstruction/config.json")
