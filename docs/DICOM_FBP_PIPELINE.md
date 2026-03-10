# DICOM + FBP Reconstruction Pipeline

## Overview

This pipeline processes DICOM CT images through FBP (Filtered Back Projection) reconstruction before feeding them to the model.

**Pipeline Flow:**
```
DICOM Files → Sinogram Generation → FBP Reconstruction → Model Training
```

## Quick Start

### 1. Install Requirements

```bash
source venv/bin/activate
pip install SimpleITK
pip install astra-toolbox
```

### 2. Prepare Your DICOM Data

Place your DICOM files in `data/Dataset/` directory:
```
data/
  └── Dataset/
      ├── 1-001.dcm
      ├── 1-002.dcm
      ├── 1-003.dcm
      └── ...
```

### 3. Test the Pipeline

```bash
python test_dicom_fbp.py
```

This will:
- Load 5 DICOM slices
- Apply FBP reconstruction
- Save visualization to `test_fbp_reconstruction.png`
- Verify everything works

### 4. Train Your Model

#### Option A: Use the example config

```bash
python train_diffusion.py --config examples/dicom_fbp_config.py
```

#### Option B: Modify existing config

Edit `configs/base_config.py` and set:
```python
data=DataConfig(
    dataset_type='dicom_fbp',  # Enable DICOM+FBP
    dicom_path='data/Dataset',
    detector_count=816,
    angle_step=0.5,
    use_fbp=True,
    cache_fbp=True,
    # ... other settings
)
```

## Configuration Parameters

### Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset_type` | Dataset type: 'bfs' or 'dicom_fbp' | 'bfs' |
| `dicom_path` | Path to DICOM directory | 'data/Dataset' |
| `detector_count` | Number of detector elements | 816 |
| `angle_step` | Angular step in degrees | 0.5 |
| `use_fbp` | Apply FBP reconstruction | True |
| `fbp_filter` | FBP filter type | 'Ram-Lak' |
| `cache_fbp` | Cache reconstructions | True |
| `cache_dir` | Cache directory | 'data/fbp_cache' |

### CT Geometry

- **detector_count=816**: Number of detector elements (width of sinogram)
- **angle_step=0.5**: Angular step → 720 projections (360°/0.5°)
- Smaller angle_step = more projections = better reconstruction quality

### FBP Filters

Available filters:
- `'Ram-Lak'`: Standard FBP filter (default)
- `'Shepp-Logan'`: Smooths high frequencies
- `'Cosine'`: Further smoothing
- `'Hamming'`: Maximum smoothing

## Dataset Splits

For a DICOM series with 133 slices:

```python
DataConfig(
    train_start=0,
    train_span=80,    # Slices 0-79 for training
    valid_start=80,
    valid_span=20,    # Slices 80-99 for validation
    test_start=100,
    test_span=33,     # Slices 100-132 for testing
)
```

## Performance Tips

### Memory Usage

- **Caching**: First run preprocesses all slices and caches them
- **Subsequent runs**: Load from cache (much faster)
- **Cache location**: `data/fbp_cache/` (can be changed)

### Speed

- Initial preprocessing: ~30-60s per slice (depends on geometry)
- With cache: ~1ms per slice
- **Tip**: Preprocess once, train many times

### GPU Memory

Adjust batch size based on reconstructed image size:
- 512×512 images: batch_size=4-8
- 256×256 images: batch_size=16-32

## File Structure

```
Experiments_P5/
├── data/
│   ├── Dataset/                   # DICOM files here
│   ├── fbp_cache/                 # Auto-generated cache
│   ├── dicom_fbp_dataset.py       # DICOM+FBP dataset class
│   └── optimized_data.py          # Main data loading
├── examples/
│   └── dicom_fbp_config.py        # Example configuration
├── test_dicom_fbp.py              # Test pipeline
└── train_diffusion.py             # Training script
```

## Troubleshooting

### Import Error: SimpleITK or ASTRA

```bash
pip install SimpleITK astra-toolbox
```

### Memory Error During Preprocessing

Reduce the number of slices:
```python
DataConfig(
    train_span=40,  # Reduce from 80
    valid_span=10   # Reduce from 20
)
```

### Slow Preprocessing

Enable caching (it's cached after first run):
```python
DataConfig(
    cache_fbp=True,
    cache_dir='data/fbp_cache'
)
```

### Poor Reconstruction Quality

Try different filters or more projections:
```python
DataConfig(
    fbp_filter='Shepp-Logan',  # Smoother
    angle_step=0.25,            # More projections (1440 instead of 720)
)
```

## Visualization

After running `test_dicom_fbp.py`, check `test_fbp_reconstruction.png`:
- **Left**: Original DICOM slice
- **Middle**: FBP reconstruction
- **Right**: Difference map

Good reconstruction should have minimal difference.

## Advanced Options

### Custom FBP Filter

Modify `dicom_fbp_dataset.py`:
```python
cfg['option'] = {
    'FilterType': 'Shepp-Logan',  # Change filter
    'FilterParameter': 0.5          # Filter parameter
}
```

### Different Geometries

Adjust in config:
```python
DataConfig(
    detector_count=1024,  # More detectors
    angle_step=0.25,       # Finer angular sampling
)
```

## Next Steps

1. ✅ Test pipeline with `python test_dicom_fbp.py`
2. ✅ Review reconstruction quality
3. ✅ Adjust parameters if needed
4. ✅ Train model with your DICOM data
5. ✅ Monitor training with WandB

---

**Questions?** Check the main README or review the code in `data/dicom_fbp_dataset.py`
