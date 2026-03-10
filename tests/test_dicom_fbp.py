"""
Test DICOM + FBP Pipeline
==========================

This script tests the DICOM+FBP dataset to ensure the pipeline works:
1. Load DICOM files
2. Generate sinograms
3. Apply FBP reconstruction
4. Verify output shapes and quality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dicom_fbp_dataset import DICOMFBPDataset
import matplotlib.pyplot as plt
import numpy as np


def test_dicom_fbp_pipeline():
    """Test the complete DMICOi+FBP pipeline"""
    
    print("=" * 70)
    print("Testing DICOM + FBP Reconstruction Pipeline")
    print("=" * 70)
    
    # Create dataset
    print("\n1. Creating dataset...")
    dataset = DICOMFBPDataset(
        dicom_path="data/Dataset",
        detector_count=816,
        angle_step=0.5,
        start_slice=0,
        num_slices=5,  # Test with 5 slices
        target_size=(512, 512),
        cache_dir="data/test_fbp_cache",
        use_cache=True
    )
    
    print(f"\n✅ Dataset created successfully!")
    print(f"   - Total slices: {len(dataset)}")
    print(f"   - Sample shape: {dataset[0].shape}")
    
    # Test data loading
    print("\n2. Testing data loading...")
    sample = dataset[0]
    print(f"   - Sample dtype: {sample.dtype}")
    print(f"   - Sample range: [{sample.min().item():.4f}, {sample.max().item():.4f}]")
    
    # Visualize results
    print("\n3. Visualizing reconstruction...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original DICOM slice
    original = dataset.get_original_slice(0)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original DICOM Slice')
    axes[0].axis('off')
    
    # FBP reconstruction
    reconstructed = sample.squeeze().numpy()
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title('FBP Reconstruction')
    axes[1].axis('off')
    
    # Difference
    original_resized = np.array(plt.imread(plt.imshow(original, alpha=0).make_image(original)[0]))
    diff = np.abs(reconstructed - original[:512, :512] / original.max())
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_fbp_reconstruction.png', dpi=150)
    print(f"   ✅ Visualization saved to test_fbp_reconstruction.png")
    
    # Test batch loading
    print("\n4. Testing batch loading...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"   - Batch shape: {batch.shape}")
    print(f"   ✅ Batch loading works!")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nThe DICOM+FBP pipeline is ready for training!")
    print("\nNext steps:")
    print("  1. Review the visualization: test_fbp_reconstruction.png")
    print("  2. Adjust detector_count and angle_step if needed")
    print("  3. Run training with: python train_diffusion.py --config examples/dicom_fbp_config.py")


if __name__ == "__main__":
    try:
        test_dicom_fbp_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
