"""
Optimized Data Loading Module for G-LED
========================================

This module provides high-performance data loading with:
- Multi-worker support
- Pin memory for fast GPU transfer
- Persistent workers
- Optional caching
- Both BFS and DICOM datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
from pathlib import Path
import warnings

try:
    import SimpleITK as sitk
    import astra
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    warnings.warn("SimpleITK or ASTRA not available. DICOM functionality disabled.")

# Import DICOM FBP dataset if available
try:
    from .dicom_fbp_dataset import DICOMFBPDataset
except (ImportError, ValueError):
    # Fallback for standalone usage
    try:
        from dicom_fbp_dataset import DICOMFBPDataset
    except ImportError:
        DICOMFBPDataset = None


class BFSDataset(Dataset):
    """
    Optimized BFS (Bluff Body Flow) Dataset.
    
    Improvements over original:
    - Memory mapping for large datasets
    - Optional preprocessing cache
    - Flexible data augmentation
    """
    
    def __init__(
        self,
        data_location: List[str],
        trajec_max_len: int = 50,
        start_n: int = 0,
        n_span: int = 510,
        use_mmap: bool = False,
        preload: bool = True
    ):
        """
        Args:
            data_location: List of .npy file paths
            trajec_max_len: Maximum trajectory length
            start_n: Starting timestep
            n_span: Number of timesteps to use
            use_mmap: Use memory mapping for large files
            preload: Preload data into memory (faster but uses RAM)
        """
        assert n_span > trajec_max_len, f"n_span ({n_span}) must be > trajec_max_len ({trajec_max_len})"
        
        self.start_n = start_n
        self.n_span = n_span
        self.trajec_max_len = trajec_max_len
        self.use_mmap = use_mmap
        
        # Load data
        if use_mmap:
            # Memory-mapped loading (doesn't load into RAM)
            # Load files individually with mmap, then slice BEFORE concatenating
            solutions = [
                np.load(loc, mmap_mode='r', allow_pickle=True)
                for loc in data_location
            ]
            # Get total length
            total_len = sum(len(s) for s in solutions)
            
            # Determine which file(s) contain our range
            if start_n + n_span <= len(solutions[0]):
                # All data in first file - just slice it
                self.solution = solutions[0][start_n:start_n + n_span]
            elif start_n >= len(solutions[0]):
                # All data in second file
                offset = start_n - len(solutions[0])
                self.solution = solutions[1][offset:offset + n_span]
            else:
                # Data spans both files - only concatenate the needed slices
                first_part = solutions[0][start_n:]
                remaining = n_span - len(first_part)
                second_part = solutions[1][:remaining]
                # Concatenate only the sliced portions (much smaller)
                self.solution = np.concatenate([first_part, second_part], axis=0)
        else:
            # Regular loading
            solutions = [
                np.load(loc, allow_pickle=True)
                for loc in data_location
            ]
            solution = np.concatenate(solutions, axis=0)
            self.solution = torch.from_numpy(solution[start_n:start_n + n_span])
            
            if preload and not torch.is_tensor(self.solution):
                self.solution = torch.from_numpy(self.solution)
    
    def __len__(self):
        return self.n_span - self.trajec_max_len
    
    def __getitem__(self, index):
        if self.use_mmap:
            # Convert to tensor on-the-fly
            return torch.from_numpy(
                np.array(self.solution[index:index + self.trajec_max_len])
            )
        else:
            return self.solution[index:index + self.trajec_max_len]


class DICOMDataset(Dataset):
    """
    Optimized DICOM/CT Dataset with sinogram generation.
    
    Improvements:
    - Index map built at init for fast __getitem__
    - Cache filenames include detector/angle params to avoid stale hits
    - Sinograms normalized to [-1, 1] before caching
    - DICOM volume only loaded when cache misses require it
    """
    
    def __init__(
        self,
        series_path: str = '../data/Dataset',
        detector_count: Optional[int] = None,
        angle_step: Optional[float] = None,
        cache_dir: Optional[str] = None,
        start_n: int = 0,
        n_samples: Optional[int] = None,
        cache_only: bool = False
    ):
        """
        Args:
            series_path: Path to DICOM series
            detector_count: Number of detectors (auto if None)
            angle_step: Angular step in degrees (default 0.25)
            cache_dir: Directory to cache processed sinograms
            start_n: Starting slice index
            n_samples: Number of samples to use (all if None)
            cache_only: If True, build index_map entirely from existing cache files
                        without loading DICOM or running ASTRA at all.  Raises
                        FileNotFoundError if expected cache files are missing.
        """
        if not DICOM_AVAILABLE:
            raise ImportError("DICOM functionality requires SimpleITK and ASTRA")
        
        self.series_path = Path(series_path)
        self.detector_count = detector_count
        self.angle_step = angle_step if angle_step is not None else 0.25
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.start_n = start_n
        self.index_map = []  # list of cache Paths (cache mode) or int indices (no-cache mode)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        det_tag = detector_count if detector_count is not None else 'auto'

        # ------------------------------------------------------------------
        # Cache-only fast path: build index_map directly from existing .npy
        # files without touching DICOM or ASTRA.
        # ------------------------------------------------------------------
        if cache_only:
            if self.cache_dir is None:
                raise ValueError("cache_only=True requires cache_dir to be set")
            import re as _re
            # Collect sinograms from ALL series in the cache directory, sorted
            # by (series_id, slice_id) to form a single flat index space.
            _pat = _re.compile(
                rf"sino_s(\d+)_i(\d+)_d{_re.escape(str(det_tag))}_a{self.angle_step:.4f}\.npy"
            )
            all_files = []
            for _f in self.cache_dir.iterdir():
                _m = _pat.match(_f.name)
                if _m:
                    all_files.append((int(_m.group(1)), int(_m.group(2)), _f))
            all_files.sort(key=lambda x: (x[0], x[1]))
            # Apply start_n / n_samples as a slice into the flat list
            subset = all_files[start_n:]
            if n_samples is not None:
                subset = subset[:n_samples]
            self.index_map = [_f for _, _, _f in subset]
            if not self.index_map:
                raise FileNotFoundError(
                    f"No cached sinograms found in {self.cache_dir} starting at "
                    f"index {start_n} (expected pattern: sino_s0_i<N>_d{det_tag}_"
                    f"a{self.angle_step:.4f}.npy).\n"
                    f"Run  python data/dicom_preprocess.py  first to generate the cache."
                )
            print(
                f"⚡ Cache-only mode: {len(self.index_map)} sinograms "
                f"loaded from {self.cache_dir} (no DICOM/ASTRA needed)"
            )
            return  # skip all DICOM loading below

        # Lazily load DICOM volume — only when a cache miss requires it
        _volume = None
        _spacing = None

        def ensure_volume():
            nonlocal _volume, _spacing
            if _volume is None:
                _volume, _spacing = self._load_series()
            return _volume, _spacing

        # Determine n_samples (may require volume if not provided)
        if n_samples is None:
            vol, _ = ensure_volume()
            n_samples = vol.shape[0] - start_n

        for i in range(n_samples):
            actual_index = start_n + i

            if self.cache_dir:
                # Naming matches dicom_preprocess.py (sino_s0_i<N>_...) so that
                # pre-generated caches are found on the first run.
                cache_path = self.cache_dir / f"sino_s0_i{actual_index}_d{det_tag}_a{self.angle_step:.4f}.npy"

                if cache_path.exists():
                    # Cache hit — bypass DICOM loading entirely
                    self.index_map.append(cache_path)
                    continue

                # Cache miss — load volume and generate sinogram
                vol, sp = ensure_volume()
                if actual_index >= vol.shape[0]:
                    break  # Requested range exceeds volume size

                ct_slice = vol[actual_index]
                dx, dy = sp[0], sp[1]
                sino = self._generate_sinogram(ct_slice, dx, dy)

                # Normalize to [-1, 1] before caching
                sino_min, sino_max = sino.min(), sino.max()
                if sino_max > sino_min:
                    sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0

                np.save(cache_path, sino.astype(np.float32))
                print(f"Cached {cache_path}")
                self.index_map.append(cache_path)

            else:
                # No cache dir — store slice indices; volume kept in memory
                vol, _ = ensure_volume()
                if actual_index >= vol.shape[0]:
                    break
                self.index_map.append(actual_index)

        # Keep volume reference only for the no-cache fallback path
        if self.cache_dir is None:
            self.volume, self.spacing = ensure_volume()

        print(f"Dataset ready: {len(self.index_map)} sinograms")

    def _load_series(self) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Load DICOM series using SimpleITK"""
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(str(self.series_path))
        reader.SetFileNames(file_names)
        img = reader.Execute()
        
        volume = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        spacing = img.GetSpacing()  # (X, Y, Z)
        
        return volume, spacing
    
    def _convert_hu_to_mu(self, ct_slice: np.ndarray) -> np.ndarray:
        """Convert HU to linear attenuation coefficient"""
        ct_slice = ct_slice.astype(np.float32, copy=False)
        mu = 0.02 * (1.0 + ct_slice / 1000.0)
        mu = np.clip(mu, 0, None)
        return mu
    
    def _generate_sinogram(
        self,
        ct_slice: np.ndarray,
        dx: float,
        dy: float
    ) -> np.ndarray:
        """Generate sinogram from CT slice using ASTRA"""
        mu_slice = self._convert_hu_to_mu(ct_slice)
        
        H, W = mu_slice.shape
        
        DSO = 1000  # Distance source to origin
        ODD = 600   # Origin to detector distance
        
        angles_deg = np.arange(0, 360, self.angle_step, dtype=np.float32)
        angles = np.deg2rad(angles_deg)
        
        vol_geom = astra.create_vol_geom(
            H, W,
            -W * dx / 2.0, W * dx / 2.0,
            -H * dy / 2.0, H * dy / 2.0
        )
        
        if self.detector_count is None:
            det_count = int(np.ceil(np.sqrt(H**2 + W**2) * 1.5))
        else:
            det_count = self.detector_count
        
        proj_geom = astra.create_proj_geom(
            'fanflat', dx, det_count,
            angles, DSO, ODD
        )
        
        projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
        
        slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)
        sid = astra.data2d.create('-vol', vol_geom, slice2d)
        sino_id, sino = astra.create_sino(sid, projector_id)
        
        astra.data2d.delete(sino_id)
        astra.data2d.delete(sid)
        astra.projector.delete(projector_id)
        
        return sino
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        entry = self.index_map[index]

        if self.cache_dir:
            # Load directly from cache file
            sino = np.load(entry)
            return torch.from_numpy(sino).unsqueeze(0).float()
        else:
            # No-cache fallback: generate on-the-fly from volume in memory
            ct_slice = self.volume[entry]
            dx, dy = self.spacing[0], self.spacing[1]
            sino = self._generate_sinogram(ct_slice, dx, dy)
            return torch.from_numpy(sino).unsqueeze(0).float()


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False
) -> DataLoader:
    """
    Create an optimized DataLoader with best practices.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        drop_last: Drop last incomplete batch
    
    Returns:
        Optimized DataLoader
    """
    # Persistent workers only work with num_workers > 0
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last
    )


def create_dataloaders_from_config(config, mode='sequential'):
    """
    Create train/valid/test dataloaders from configuration.
    
    Args:
        config: ExperimentConfig object
        mode: 'sequential', 'diffusion', or 'dicom_fbp'
    
    Returns:
        Dictionary with 'train', 'valid', 'test' DataLoaders
    """
    # Determine dataset type based on mode or config
    dataset_type = getattr(config.data, 'dataset_type', 'bfs')
    
    # Create datasets based on type
    if dataset_type == 'dicom_fbp' or mode == 'dicom_fbp':
        # DICOM + FBP reconstruction pipeline
        if not DICOM_AVAILABLE:
            raise ImportError("DICOM+FBP mode requires SimpleITK and ASTRA. Install: pip install SimpleITK pyastra")
        
        if DICOMFBPDataset is None:
            raise ImportError("DICOMFBPDataset not available. Check dicom_fbp_dataset.py")
        
        print(f"📊 Using DICOM+FBP pipeline: {config.data.dicom_path}")
        
        train_dataset = DICOMFBPDataset(
            dicom_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            start_slice=config.data.train_start,
            num_slices=config.data.train_span,
            target_size=(config.data.image_height, config.data.image_width),
            cache_dir=config.data.cache_dir if config.data.cache_fbp else None,
            use_cache=config.data.cache_fbp
        )
        
        valid_dataset = DICOMFBPDataset(
            dicom_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            start_slice=config.data.valid_start,
            num_slices=config.data.valid_span,
            target_size=(config.data.image_height, config.data.image_width),
            cache_dir=config.data.cache_dir if config.data.cache_fbp else None,
            use_cache=config.data.cache_fbp
        )
        
        test_dataset = DICOMFBPDataset(
            dicom_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            start_slice=config.data.test_start,
            num_slices=config.data.test_span,
            target_size=(config.data.image_height, config.data.image_width),
            cache_dir=config.data.cache_dir if config.data.cache_fbp else None,
            use_cache=config.data.cache_fbp
        )
    
    elif mode == 'diffusion' or mode == 'sequential' or dataset_type == 'bfs':
        # BFS dataset for sequential and diffusion training
        # 'diffusion' mode trains the spatial diffusion model on the same BFS sinogram data
        print(f"📊 Using BFS dataset (mode={mode})")
        
        train_dataset = BFSDataset(
            data_location=config.data.data_location,
            trajec_max_len=config.data.seq_length,
            start_n=config.data.train_start,
            n_span=config.data.train_span,
            use_mmap=True,  # Use memory mapping to avoid RAM overflow
            preload=False
        )
        
        valid_dataset = BFSDataset(
            data_location=config.data.data_location,
            trajec_max_len=config.data.seq_length_valid,
            start_n=config.data.valid_start,
            n_span=config.data.valid_span,
            use_mmap=True,  # Use memory mapping to avoid RAM overflow
            preload=False
        )
        
        test_dataset = BFSDataset(
            data_location=config.data.data_location,
            trajec_max_len=config.data.seq_length_valid,
            start_n=config.data.test_start,
            n_span=config.data.test_span,
            use_mmap=True,  # Use memory mapping to avoid RAM overflow
            preload=False
        )
    
    elif mode == 'dicom_sino' or dataset_type == 'dicom_sino':
        # Raw sinogram pipeline for sparse-view CT reconstruction.
        # DICOMDataset generates full sinograms via ASTRA forward projection;
        # the diffusion trainer builds masked conditioning on-the-fly.
        #
        # FAST PATH: if sino_cache_dir already contains .npy files produced
        # by dicom_preprocess.py, use cache_only=True to skip all DICOM/ASTRA
        # work entirely — just load the .npy files directly.
        _sino_cache = getattr(config.data, 'sino_cache_dir', 'data/sino_cache')
        _cache_ready = (
            _sino_cache
            and os.path.isdir(_sino_cache)
            and any(f.endswith('.npy') for f in os.listdir(_sino_cache))
        )
        if _cache_ready:
            print(f"⚡ sino_cache detected at '{_sino_cache}' — skipping DICOM loading")
        else:
            if not DICOM_AVAILABLE:
                raise ImportError(
                    "dicom_sino mode requires SimpleITK and ASTRA (no pre-built cache found). "
                    "Install SimpleITK with: pip install SimpleITK; "
                    "follow ASTRA Toolbox installation docs for GPU support. "
                    "Or run  python data/dicom_preprocess.py  to pre-generate the cache."
                )
            print(f"📊 Using DICOM sinogram pipeline (mode={mode}) — will build cache")

        train_dataset = DICOMDataset(
            series_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            cache_dir=_sino_cache,
            start_n=config.data.train_start,
            n_samples=config.data.train_span,
            cache_only=_cache_ready,
        )

        valid_dataset = DICOMDataset(
            series_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            cache_dir=_sino_cache,
            start_n=config.data.valid_start,
            n_samples=config.data.valid_span,
            cache_only=_cache_ready,
        )

        test_dataset = DICOMDataset(
            series_path=config.data.dicom_path,
            detector_count=config.data.detector_count,
            angle_step=config.data.angle_step,
            cache_dir=_sino_cache,
            start_n=config.data.test_start,
            n_samples=config.data.test_span,
            cache_only=_cache_ready,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}, dataset_type: {dataset_type}")
    
    # Create dataloaders
    train_loader = create_optimized_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=False
    )
    
    valid_loader = create_optimized_dataloader(
        valid_dataset,
        batch_size=config.training.batch_size_valid,
        shuffle=False,  # Don't shuffle validation
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=False
    )
    
    test_loader = create_optimized_dataloader(
        test_dataset,
        batch_size=config.training.batch_size_valid,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
        drop_last=False
    )
    
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    print("Testing Data Loading Module\n")
    
    # Test BFS dataset
    print("Testing BFS Dataset...")
    try:
        dataset = BFSDataset(
            data_location=['../data/data0.npy'],
            trajec_max_len=41,
            start_n=0,
            n_span=100
        )
        print(f"✅ BFS Dataset: {len(dataset)} samples")
        print(f"   Sample shape: {dataset[0].shape}")
    except Exception as e:
        print(f"❌ BFS Dataset failed: {e}")
    
    # Test optimized dataloader
    print("\nTesting Optimized DataLoader...")
    try:
        loader = create_optimized_dataloader(
            dataset,
            batch_size=4,
            num_workers=2
        )
        batch = next(iter(loader))
        print(f"✅ DataLoader: batch shape {batch.shape}")
    except Exception as e:
        print(f"❌ DataLoader failed: {e}")
