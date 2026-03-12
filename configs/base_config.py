"""
Base Configuration System for G-LED Experiments
================================================

This provides a unified configuration system for all experiments,
allowing easy switching between different setups without code changes.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
from datetime import datetime


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Dataset type: 'bfs' for BFS simulation, 'dicom_fbp' for DICOM with FBP
    dataset_type: str = 'bfs'  # or 'dicom_fbp'
    
    # Dataset paths - UPDATE THESE FOR YOUR CLUSTER!
    # For BFS: .npy files
    # For DICOM: path to DICOM directory
    data_location: List[str] = field(default_factory=lambda: ['data/data0.npy', 'data/data1.npy'])
    dicom_path: str = 'data/Dataset'  # Path to DICOM series
    
    # Dataset splits
    train_start: int = 0
    train_span: int = 8000
    valid_start: int = 8000
    valid_span: int = 500
    test_start: int = 9500
    test_span: int = 512
    
    # Sequence parameters (for BFS sequential model)
    seq_length: int = 41  # Trajectory max length
    seq_length_valid: int = 450
    
    # Image dimensions (for CT/medical imaging)
    image_height: int = 512
    image_width: int = 256
    
    # Detector configuration (for CT/DICOM)
    detector_count: int = 816  # Number of detector elements
    angle_step: float = 0.5  # Angular step in degrees (e.g., 0.5 for 720 projections)

    # Sparse-view split strategy (governs how condition / target are split in the diffusion trainer)
    # Options:
    #   'angle_25pct'         – every 4th row is condition, rest (75 %) are target        [0::4]
    #   'angle_12pct'         – every 8th row is condition, rest (87.5 %) are target      [0::8]
    #   'angle_block_50pct'   – top half is condition, bottom half is target               [H//2]
    #   'angle_step_50pct'    – every other row is condition, odd rows are target          [0::2]
    #   'angle_limited_25pct' – every 4th row in the first 75 % of rows is condition
    sparse_strategy: str = 'angle_25pct'

    # FBP reconstruction settings
    use_fbp: bool = True  # Apply FBP reconstruction
    fbp_filter: str = 'Ram-Lak'  # FBP filter type: Ram-Lak, Shepp-Logan, Cosine, etc.
    cache_fbp: bool = True  # Cache FBP reconstructions
    cache_dir: Optional[str] = 'data/fbp_cache'  # FBP cache directory (for dicom_fbp mode)
    sino_cache_dir: str = 'data/sino_cache'  # Sinogram cache directory (for dicom_sino mode)

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Sequential Transformer
    n_layer: int = 8  # Number of transformer blocks
    n_ctx: int = 40   # Context window
    n_embd: int = 2048  # Embedding dimension
    n_head: int = 4   # Attention heads
    
    # UNet for diffusion
    unet_dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8)
    memory_efficient: bool = True
    
    # Downsampling
    coarse_dim: Tuple[int, int] = (32, 32)
    coarse_mode: str = 'bilinear'
    
    # Model features
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    activation_function: str = 'relu'
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    
    # Diffusion specific
    num_sample_steps: int = 20
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 80.0
    S_tmin: float = 0.05
    S_tmax: float = 50.0
    S_noise: float = 1.003
    cond_drop_prob: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True  # Use deterministic algorithms (slower but reproducible)
    
    # Experiment tracking
    use_wandb: bool = True  # Enable Weights & Biases logging
    wandb_project: str = "experimentp5-gled"  # Base project name
    wandb_entity: Optional[str] = None  # Your wandb username/team (None = default)
    
    # Batch sizes
    batch_size: int = 16
    batch_size_valid: int = 16
    gradient_accumulation_steps: int = 1
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    use_fused_optimizer: bool = True  # Fused AdamW
    
    # Learning rate schedule
    warmup_steps: Optional[int] = 1000
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'constant'
    gamma: float = 0.99083194489  # For StepLR
    cosine_max_steps: Optional[int] = None
    
    # Regularization
    max_grad_norm: float = 1.0
    grad_clip_enabled: bool = True
    
    # Training duration
    epoch_num: int = 100
    
    # Curriculum learning (for sequential model)
    start_Nt: int = 1
    d_Nt: int = 1
    march_tol: float = 0.01
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    precision: str = 'fp16'  # 'fp16', 'bf16', or 'fp32'
    
    # Distributed training
    use_multi_gpu: bool = False
    world_size: int = 1
    
    # Checkpointing
    save_every: int = 1
    keep_last_n_checkpoints: int = 5
    
    # Logging
    log_every: int = 10
    eval_every: int = 1
    
    # Resume training
    resume_from: Optional[str] = None
    
    # Device
    device: str = 'cuda:0'


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs"""
    name: str = 'default_experiment'
    description: str = ''
    mode: str = 'sequential'  # 'sequential' or 'diffusion'
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Output paths (auto-generated)
    output_dir: str = ''
    checkpoint_dir: str = ''
    log_dir: str = ''
    
    def __post_init__(self):
        """Setup output directories"""
        if not self.output_dir:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'./output/{self.name}/run_{timestamp}'
        
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        # Directories are created lazily (set_output_dir or first save/train call)
        # to avoid ghost directories when --output_dir overrides the auto path.
    
    def set_output_dir(self, output_dir: str):
        """Override output directory and re-derive checkpoint/log paths"""
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def save(self, path: Optional[str] = None):
        """Save configuration to JSON"""
        if path is None:
            path = os.path.join(self.log_dir, 'config.json')
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Convert to dict
        config_dict = {
            'name': self.name,
            'description': self.description,
            'mode': self.mode,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(
            name=config_dict['name'],
            description=config_dict['description'],
            mode=config_dict['mode']
        )
        
        # Load sub-configs
        config.data = DataConfig(**config_dict['data'])
        config.model = ModelConfig(**config_dict['model'])
        config.training = TrainingConfig(**config_dict['training'])
        
        return config
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 70)
        print(f"Experiment: {self.name}")
        print(f"Mode: {self.mode}")
        if self.description:
            print(f"Description: {self.description}")
        print("=" * 70)
        print("\n📊 Data Configuration:")
        print(f"  Train: {self.data.train_start} → {self.data.train_start + self.data.train_span}")
        print(f"  Valid: {self.data.valid_start} → {self.data.valid_start + self.data.valid_span}")
        print(f"  Image Size: {self.data.image_height}×{self.data.image_width}")
        if self.data.detector_count:
            print(f"  Detector: {self.data.detector_count} (angle: {self.data.angle_step})")
        
        print("\n🏗️  Model Configuration:")
        print(f"  Transformer: {self.model.n_layer} layers, {self.model.n_embd} dim")
        print(f"  UNet: dim={self.model.unet_dim}, mults={self.model.dim_mults}")
        print(f"  Coarse: {self.model.coarse_dim}")
        
        print("\n⚙️  Training Configuration:")
        print(f"  Batch Size: {self.training.batch_size} (accum={self.training.gradient_accumulation_steps})")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Precision: {self.training.precision}")
        print(f"  Multi-GPU: {self.training.use_multi_gpu}")
        print(f"  Epochs: {self.training.epoch_num}")
        
        print("\n📁 Output:")
        print(f"  Directory: {self.output_dir}")
        print("=" * 70 + "\n")


def get_preset_config(preset_name: str) -> ExperimentConfig:
    """
    Get predefined experiment configurations.
    
    Presets based on different branches:
    - 'baseline': Original main branch configuration
    - '720_816_standard': Standard 720×816 setup
    - '720_816_high_unet': High capacity UNet (dim=128)
    - '720_816_medium_unet': Medium capacity (dim=64)
    - 'updated_1024': Highest resolution (1024×512)
    - 'feather_720_432': Feather aspect ratio
    """
    # not needed
    if preset_name == 'baseline':
        return ExperimentConfig(
            name='baseline_main_branch',
            description='Original main branch configuration — frozen reference, do not modify',
            data=DataConfig(
                image_height=512,
                image_width=256
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4)
            ),
            training=TrainingConfig(
                batch_size=1,
                epoch_num=50,
                use_amp=False  # Original didn't use AMP
            )
        )
    # not needed
    elif preset_name == 'baseline_optimized':
        return ExperimentConfig(
            name='baseline_optimized',
            description='Baseline architecture with modern training settings (batch=8, AMP). '
                        'Same model size as baseline but compute-matched to other presets.',
            data=DataConfig(
                image_height=512,
                image_width=256
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4)  # Same small model as baseline
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True,
                gradient_accumulation_steps=1  # No accumulation needed at this size
            )
        )
    
    elif preset_name == '720_816_standard':
        return ExperimentConfig(
            name='720_816_standard',
            description='Standard 720×816 configuration with optimizations',
            data=DataConfig(
                image_height=720,  # full sinogram height — masked conditioning keeps cond and target the same size
                image_width=816,
                detector_count=816,
                angle_step=360/720
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )
    
    elif preset_name == '720_816_high_unet':
        return ExperimentConfig(
            name='720_816_high_capacity',
            description='High capacity UNet (dim=128) for better quality',
            data=DataConfig(
                image_height=720,  # full sinogram height — masked conditioning keeps cond and target the same size
                image_width=816,
                detector_count=816,
                angle_step=360/720
            ),
            model=ModelConfig(
                unet_dim=128,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=4,  # Smaller due to memory
                epoch_num=500,
                use_amp=True,
                gradient_accumulation_steps=2  # Effective batch = 8
            )
        )
    
    elif preset_name == '720_816_medium_unet':
        return ExperimentConfig(
            name='720_816_medium_capacity',
            description='Medium capacity UNet (dim=64) - balanced',
            data=DataConfig(
                image_height=720,  # full sinogram height — masked conditioning keeps cond and target the same size
                image_width=816,
                detector_count=816,
                angle_step=360/720
            ),
            model=ModelConfig(
                unet_dim=64,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )
    # not needed
    elif preset_name == 'updated_1024':
        return ExperimentConfig(
            name='updated_1024_512',
            description='Highest resolution (1024×512) configuration',
            data=DataConfig(
                image_height=1024,  # full sinogram height (1024 angles, 1024%16=0)
                image_width=512,
                detector_count=1024,
                angle_step=360/1024,
                train_start=132  # As in Updated-Code branch
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=4,  # Lower due to high resolution
                epoch_num=500,
                use_amp=True,
                gradient_accumulation_steps=2
            )
        )
    # not needed
    elif preset_name == 'feather_720_432':
        return ExperimentConfig(
            name='feather_aspect_ratio',
            description='Feather aspect ratio experiment (720×448)',
            data=DataConfig(
                image_height=720,  # full sinogram height — masked conditioning keeps cond and target the same size
                image_width=448,
                detector_count=816,
                angle_step=360/720
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )
    
    elif preset_name == 'optimized_default':
        return ExperimentConfig(
            name='optimized_default',
            description='Recommended default with all optimizations',
            data=DataConfig(
                image_height=720,  # full sinogram height — masked conditioning keeps cond and target the same size
                image_width=816,
                detector_count=816,
                angle_step=360/720,
                num_workers=8  # Maximize data loading
            ),
            model=ModelConfig(
                unet_dim=64,  # Good balance
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True,
                use_multi_gpu=True, 
                use_fused_optimizer=True,
                max_grad_norm=1.0
            )
        )
    
    elif preset_name == '720_816_12pct':
        return ExperimentConfig(
            name='720_816_12.5pct_sparse',
            description='12.5 % sparse-view CT: every 8th projection angle as condition '
                        '(90/720 rows); 630 target rows padded to 640. '
                        'Matches the 720_816_12.5% G-LED branch.',
            data=DataConfig(
                image_height=720,   # full sinogram height (720 angles, 720%16=0)
                image_width=816,
                detector_count=816,
                angle_step=360/720,
                sparse_strategy='angle_12pct'
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )
    # not needed
    elif preset_name == '720_816_angle_block_50':
        return ExperimentConfig(
            name='720_816_angle_block_50pct',
            description='50 % sparse-view CT — contiguous block split: top 360 rows are '
                        'the known (condition) projections; bottom 360 rows are the target; '
                        'padded to 368. Matches the 720_816_model_angle_block G-LED branch.',
            data=DataConfig(
                image_height=720,   # full sinogram height (720 angles, 720%16=0)
                image_width=816,
                detector_count=816,
                angle_step=360/720,
                sparse_strategy='angle_block_50pct'
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )

    elif preset_name == '720_816_angle_step_50':
        return ExperimentConfig(
            name='720_816_angle_step_50pct',
            description='50 % sparse-view CT — interleaved step split: even-indexed rows '
                        'are the known (condition) projections; odd-indexed rows are the '
                        'target; 360 target rows padded to 368. '
                        'Matches the 720_816_model_angle_step G-LED branch.',
            data=DataConfig(
                image_height=720,   # full sinogram height (720 angles, 720%16=0)
                image_width=816,
                detector_count=816,
                angle_step=360/720,
                sparse_strategy='angle_step_50pct'
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )

    elif preset_name == '720_816_limited_view':
        return ExperimentConfig(
            name='720_816_limited_view_25pct',
            description='Limited-view sparse CT: every 4th row within the first 75 % of '
                        'projection angles is the condition (135 rows); the remaining 585 '
                        'rows are the target, padded to 592. '
                        'Matches the 720_816_25%_limited_view G-LED branch.',
            data=DataConfig(
                image_height=720,   # full sinogram height (720 angles, 720%16=0)
                image_width=816,
                detector_count=816,
                angle_step=360/720,
                sparse_strategy='angle_limited_25pct'
            ),
            model=ModelConfig(
                unet_dim=32,
                dim_mults=(1, 2, 4, 8)
            ),
            training=TrainingConfig(
                batch_size=8,
                epoch_num=500,
                use_amp=True
            )
        )

    else:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: baseline, baseline_optimized, 720_816_standard, 720_816_high_unet, "
            f"720_816_medium_unet, updated_1024, feather_720_432, optimized_default, "
            f"720_816_12pct, 720_816_angle_block_50, 720_816_angle_step_50, 720_816_limited_view"
        )


if __name__ == '__main__':
    # Test configurations
    print("\nTesting Configuration System\n")
    
    # Create and print optimized default
    config = get_preset_config('optimized_default')
    config.print_summary()
    
    # Save config
    config.save('test_config.json')
    
    # Load config
    loaded = ExperimentConfig.load('test_config.json')
    print("\n✅ Configuration system working!")
    
    # Show all presets
    print("\n📋 Available Presets:")
    presets = [
        'baseline', '720_816_standard', '720_816_high_unet',
        '720_816_medium_unet', 'updated_1024', 'feather_720_432',
        'optimized_default'
    ]
    for preset in presets:
        cfg = get_preset_config(preset)
        print(f"  - {preset}: {cfg.description}")
