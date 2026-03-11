"""
DICOM Dataset with FBP Reconstruction Pipeline
===============================================

Pipeline: DICOM → Sinogram → FBP Reconstruction → Model

This dataset:
1. Loads DICOM slices
2. Generates sinograms (forward projection)
3. Applies FBP reconstruction
4. Feeds reconstructed images to the model
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from tqdm import tqdm
import warnings

try:
    import SimpleITK as sitk  # type: ignore[import-untyped]
    import astra  # type: ignore[import-untyped]
    RECONSTRUCTION_AVAILABLE = True
except ImportError:
    RECONSTRUCTION_AVAILABLE = False
    warnings.warn("SimpleITK or ASTRA not available. DICOM+FBP functionality disabled.")


class DICOMFBPDataset(Dataset):
    """
    Dataset that processes DICOM files through FBP reconstruction.
    
    Pipeline:
        DICOM slice → Sinogram (forward projection) → FBP → Reconstructed image
    
    Args:
        dicom_path: Path to DICOM series directory
        detector_count: Number of detector elements
        angle_step: Angular step in degrees (default: 0.25)
        start_slice: Starting slice index
        num_slices: Number of slices to process (None = all)
        target_size: Resize reconstructed images to (H, W)
        cache_dir: Directory to cache processed data
        use_cache: Whether to use/save cached data
    """
    
    def __init__(
        self,
        dicom_path: str = "data/Dataset",
        detector_count: int = 816,
        angle_step: float = 0.5,
        start_slice: int = 0,
        num_slices: Optional[int] = None,
        target_size: Tuple[int, int] = (512, 512),
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        if not RECONSTRUCTION_AVAILABLE:
            raise ImportError("DICOMFBPDataset requires SimpleITK and ASTRA")
        
        self.dicom_path = Path(dicom_path)
        self.detector_count = detector_count
        self.angle_step = angle_step
        self.start_slice = start_slice
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        # Load DICOM volume
        print(f"📁 Loading DICOM series from {dicom_path}...")
        self.volume, self.spacing = self._load_dicom_series()
        
        # Determine number of slices to process
        total_slices = self.volume.shape[0]
        if num_slices is None:
            num_slices = total_slices - start_slice
        self.num_slices = max(0, min(num_slices, total_slices - start_slice))
        
        print(f"✅ Loaded {total_slices} slices, processing {self.num_slices} slices")
        print(f"   Spacing: {self.spacing} mm")
        print(f"   Volume shape: {self.volume.shape}")
        
        # Setup cache
        if self.cache_dir and self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"💾 Cache enabled: {self.cache_dir}")
        
        # Preprocess all slices
        print("🔄 Preprocessing slices...")
        self.reconstructed_images = self._preprocess_all_slices()
        print(f"✅ Preprocessing complete! Dataset size: {len(self)}")
    
    def _load_dicom_series(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Load ALL DICOM series from directory and concatenate along slice axis"""
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(self.dicom_path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {self.dicom_path}")
        
        volumes = []
        spacing = None
        for series_uid in sorted(series_ids):
            file_names = reader.GetGDCMSeriesFileNames(str(self.dicom_path), series_uid)
            reader.SetFileNames(file_names)
            img = reader.Execute()
            volumes.append(sitk.GetArrayFromImage(img))  # (Z, Y, X)
            if spacing is None:
                spacing = img.GetSpacing()  # use spacing from first series
        
        volume = np.concatenate(volumes, axis=0)  # (total_Z, Y, X)
        print(f"   Loaded {len(series_ids)} series → {volume.shape[0]} total slices")
        return volume, spacing
    
    def _convert_hu_to_mu(self, ct_slice: np.ndarray) -> np.ndarray:
        """Convert HU to linear attenuation coefficient"""
        ct_slice = ct_slice.astype(np.float32, copy=False)
        # mu = mu_water * (1 + HU/1000), mu_water ≈ 0.02 mm^-1
        mu = 0.02 * (1.0 + ct_slice / 1000.0)
        mu = np.clip(mu, 0, None)  # Attenuation can't be negative
        return mu
    
    def _generate_sinogram(self, ct_slice: np.ndarray) -> np.ndarray:
        """Generate sinogram from CT slice using ASTRA"""
        # Convert HU to attenuation
        mu_slice = self._convert_hu_to_mu(ct_slice)
        H, W = mu_slice.shape
        dx, dy, dz = self.spacing
        
        # Geometry parameters (typical CT geometry)
        DSO = 1000.0  # Distance source to origin (mm)
        ODD = 600.0   # Origin to detector distance (mm)
        
        # Projection angles
        angles_deg = np.arange(0, 360, self.angle_step, dtype=np.float32)
        angles_rad = np.deg2rad(angles_deg)
        
        # Volume geometry
        vol_geom = astra.create_vol_geom(
            H, W,
            -W * dx / 2.0, W * dx / 2.0,
            -H * dy / 2.0, H * dy / 2.0
        )
        
        # Projection geometry (fan beam)
        det_spacing = dx
        proj_geom = astra.create_proj_geom(
            'fanflat', det_spacing, self.detector_count,
            angles_rad, DSO, ODD
        )
        
        # Forward projection (GPU)
        slice_data = np.ascontiguousarray(mu_slice, dtype=np.float32)
        vol_id = astra.data2d.create('-vol', vol_geom, slice_data)
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        sino_id, sinogram = astra.create_sino(vol_id, projector_id)
        
        # Cleanup
        astra.data2d.delete(sino_id)
        astra.data2d.delete(vol_id)
        astra.projector.delete(projector_id)
        
        return sinogram
    
    def _fbp_reconstruction(self, sinogram: np.ndarray) -> np.ndarray:
        """Apply FBP (Filtered Back Projection) reconstruction"""
        H, W = self.volume.shape[1], self.volume.shape[2]
        dx, dy, dz = self.spacing
        
        # Same geometry as forward projection
        DSO = 1000.0
        ODD = 600.0
        angles_deg = np.arange(0, 360, self.angle_step, dtype=np.float32)
        angles_rad = np.deg2rad(angles_deg)
        
        vol_geom = astra.create_vol_geom(
            H, W,
            -W * dx / 2.0, W * dx / 2.0,
            -H * dy / 2.0, H * dy / 2.0
        )
        
        det_spacing = dx
        proj_geom = astra.create_proj_geom(
            'fanflat', det_spacing, self.detector_count,
            angles_rad, DSO, ODD
        )
        
        # Create reconstruction buffers
        rec_id = astra.data2d.create('-vol', vol_geom)
        sino_id = astra.data2d.create('-sino', proj_geom, sinogram)
        projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
        
        # Configure FBP_CUDA algorithm (supports fan beam on GPU)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sino_id
        cfg['ProjectorId'] = projector_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}  # Standard ramp filter
        
        # Run reconstruction
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        # Get result
        reconstruction = astra.data2d.get(rec_id)
        
        # Cleanup
        astra.algorithm.delete(alg_id)
        astra.projector.delete(projector_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        
        return reconstruction
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size using torch"""
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(
            img_tensor, size=self.target_size, mode='bilinear', align_corners=False
        )
        return resized.squeeze(0).squeeze(0).numpy()
    
    def _process_single_slice(self, slice_idx: int) -> np.ndarray:
        """Process a single slice through the full pipeline"""
        cache_file = None
        cache_file = None
        if self.cache_dir and self.use_cache:
            cache_file = self.cache_dir / f"fbp_slice_{slice_idx:04d}.npy"
            if cache_file.exists():
                return np.load(cache_file)
        
        # Get slice
        actual_idx = self.start_slice + slice_idx
        ct_slice = self.volume[actual_idx]
        
        # Pipeline: DICOM → Sinogram → FBP → Normalize → Resize
        sinogram = self._generate_sinogram(ct_slice)
        reconstructed = self._fbp_reconstruction(sinogram)
        normalized = self._normalize_image(reconstructed)
        resized = self._resize_image(normalized)
        
        # Cache if enabled
        if cache_file:
            np.save(cache_file, resized)
        
        return resized
    
    def _preprocess_all_slices(self) -> np.ndarray:
        """Preprocess all slices in the dataset"""
        images = []
        for i in tqdm(range(self.num_slices), desc="Processing slices"):
            img = self._process_single_slice(i)
            images.append(img)
        
        return np.stack(images, axis=0)  # (N, H, W)
    
    def __len__(self) -> int:
        return self.num_slices
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Reconstructed image tensor of shape (1, H, W)
        """
        img = self.reconstructed_images[idx]
        return torch.from_numpy(img).unsqueeze(0).float()  # Add channel dim
    
    def get_original_slice(self, idx: int) -> np.ndarray:
        """Get original DICOM slice for comparison"""
        actual_idx = self.start_slice + idx
        return self.volume[actual_idx]


# Test function
if __name__ == "__main__":
    # Test the dataset
    dataset = DICOMFBPDataset(
        dicom_path="data/Dataset",
        detector_count=816,
        angle_step=0.5,
        start_slice=0,
        num_slices=10,  # Test with 10 slices
        target_size=(512, 512),
        cache_dir="data/fbp_cache"
    )
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Sample shape: {dataset[0].shape}")
    
    # Save a sample
    sample = dataset[0].squeeze().numpy()
    import matplotlib.pyplot as plt
    plt.imsave("fbp_sample.png", sample, cmap='gray')
    print("✅ Sample saved to fbp_sample.png")
