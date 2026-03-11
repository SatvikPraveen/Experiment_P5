"""
Example Configuration for DICOM + FBP Training
===============================================

This configuration demonstrates how to train a model on DICOM data
with FBP reconstruction in the pipeline.

Pipeline: DICOM → Sinogram → FBP Reconstruction → Model
"""

from configs.base_config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig


def get_dicom_fbp_config():
    """Get configuration for DICOM+FBP training"""
    return ExperimentConfig(
        name='dicom_fbp_example',
        description='DICOM with FBP reconstruction pipeline',
        mode='dicom_fbp',
        
        data=DataConfig(
            # Dataset type
            dataset_type='dicom_fbp',  # Use DICOM+FBP pipeline
            
            # DICOM data paths
            dicom_path='data/Dataset',  # Path to DICOM files
            
            # Dataset splits (in number of slices)
            # Sized for a ~133-slice DICOM volume
            train_start=0,
            train_span=100,  # First 100 slices for training
            valid_start=100,
            valid_span=20,   # Next 20 for validation
            test_start=120,
            test_span=13,    # Remaining for testing
            
            # Image dimensions
            image_height=512,
            image_width=512,
            
            # CT geometry
            detector_count=816,  # Number of detector elements
            angle_step=0.5,      # 0.5° step → 720 projections
            
            # FBP settings
            use_fbp=True,
            fbp_filter='Ram-Lak',  # Standard FBP filter
            cache_fbp=True,        # Cache reconstructions
            cache_dir='data/fbp_cache',
            
            # Data loading
            num_workers=4,
            pin_memory=True
        ),
        
        model=ModelConfig(
            # Adjust based on your model architecture
            unet_dim=64,
            dim_mults=(1, 2, 4, 8),
            memory_efficient=True
        ),
        
        training=TrainingConfig(
            batch_size=4,      # Adjust based on GPU memory
            epoch_num=100,
            learning_rate=1e-4,
            use_amp=True,      # Mixed precision
            use_wandb=True,
            wandb_project='dicom-fbp-reconstruction'
        )
    )


if __name__ == "__main__":
    config = get_dicom_fbp_config()
    config.print_summary()
    config.save('configs/dicom_fbp_config.json')
    print("\n✅ Configuration saved to configs/dicom_fbp_config.json")
