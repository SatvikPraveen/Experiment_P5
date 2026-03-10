# Data Splits and Repository Structure

## Data Splits: Train / Validation / Test ✅

**Yes**, the repository has full train/validation/test split support implemented across both sequential and diffusion training modes.

### Split Configuration

All splits are defined in [configs/base_config.py](configs/base_config.py) under `DataConfig`:

```python
# Default splits (can be overridden per preset)
train_start: int = 0
train_span: int = 8000       # 8000 samples for training
valid_start: int = 8000
valid_span: int = 500        # 500 samples for validation
test_start: int = 9500
test_span: int = 512         # 512 samples for testing
```

**Key characteristics**:
- **Non-overlapping**: Train ends at 8000, validation starts at 8000
- **Sequential indexing**: Uses timestep ranges from .npy files
- **Configurable**: Each preset can override these values

### How Splits Work

#### 1. BFS Dataset (Sequential Training)
Located in [data/optimized_data.py](data/optimized_data.py):

```python
train_dataset = BFSDataset(
    data_location=config.data.data_location,
    trajec_max_len=config.data.seq_length,
    start_n=config.data.train_start,      # 0
    n_span=config.data.train_span,        # 8000
    ...
)

valid_dataset = BFSDataset(
    data_location=config.data.data_location,
    trajec_max_len=config.data.seq_length_valid,
    start_n=config.data.valid_start,      # 8000
    n_span=config.data.valid_span,        # 500
    ...
)

test_dataset = BFSDataset(
    data_location=config.data.data_location,
    trajec_max_len=config.data.seq_length_valid,
    start_n=config.data.test_start,       # 9500
    n_span=config.data.test_span,         # 512
    ...
)
```

#### 2. DICOM Dataset (Diffusion Training)
Same split logic applies:

```python
train_dataset = DICOMDataset(
    start_n=config.data.train_start,
    n_samples=config.data.train_span
)

valid_dataset = DICOMDataset(
    start_n=config.data.valid_start,
    n_samples=config.data.valid_span
)

test_dataset = DICOMDataset(
    start_n=config.data.test_start,
    n_samples=config.data.test_span
)
```

### Split Usage in Training

#### Sequential Training
In [train_sequential.py](train_sequential.py):
```python
dataloaders = create_dataloaders_from_config(config, mode='sequential')
train_loader = dataloaders['train']
valid_loader = dataloaders['valid']  # ✅ Validation used

# Training loop validates every epoch
trainer.train_epochs(
    train_loader=train_loader,
    valid_loader=valid_loader,  # ✅ Passed to trainer
    num_epochs=config.training.num_epochs
)
```

In [train_test_seq/optimized_trainer.py](train_test_seq/optimized_trainer.py):
```python
# Validation happens during training
if (epoch + 1) % validation_interval == 0:
    valid_loss = self.validate(self.valid_loader)  # ✅ Validates
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "valid_loss": valid_loss,  # ✅ Logged
        "learning_rate": current_lr
    })
```

#### Diffusion Training
In [train_diffusion.py](train_diffusion.py):
```python
dataloaders = create_dataloaders_from_config(config, mode='diffusion')
train_loader = dataloaders['train']
valid_loader = dataloaders['valid']  # ✅ Validation available

trainer = DiffusionTrainer(...)
trainer.train_epochs(
    train_loader=train_loader,
    valid_loader=valid_loader,  # ✅ Can be used
    ...
)
```

### Testing (Evaluation Scripts)

#### Sequential Model Evaluation
[eval_sequential.py](eval_sequential.py):
```python
dataloaders = create_dataloaders_from_config(config, mode='sequential')

if args.split == 'valid':
    dataloader = dataloaders['valid']  # ✅ Validation set
elif args.split == 'train':
    dataloader = dataloaders['train']   # ✅ Training set
else:
    dataloader = dataloaders['test']    # ✅ Test set

# Evaluate on chosen split
metrics = evaluate(model, dataloader, device)
```

#### Diffusion Model Evaluation
[eval_diffusion.py](eval_diffusion.py):
```python
dataloaders = create_dataloaders_from_config(config, mode='diffusion')
# Same split selection logic
```

### Split Sizes by Preset

Most presets use default split configuration:
- **Train**: 8000 samples (timesteps 0-7999)
- **Valid**: 500 samples (timesteps 8000-8499)
- **Test**: 512 samples (timesteps 9500-10011)

**Exception**: `updated_1024` preset (from Updated-Code branch):
```python
train_start=132  # Starts at timestep 132 instead of 0
train_span=8000
```

### Data Loading Optimization

All splits use the optimized dataloader from [data/optimized_data.py](data/optimized_data.py):
```python
create_optimized_dataloader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,              # True for train, False for valid/test
    num_workers=4,             # Parallel data loading
    pin_memory=True,           # Fast GPU transfer
    persistent_workers=True,   # Keep workers alive
    prefetch_factor=2,         # Prefetch batches
    drop_last=False            # Keep all samples
)
```

**Performance features**:
- Multi-worker parallel loading
- Memory pinning for GPU transfer
- Persistent workers across epochs
- Batch prefetching
- Optional memory mapping for large datasets

---

## Repository Structure: ExperimentP5/ Folder

### Why is there an `ExperimentP5/` folder inside the workspace?

Looking at the repository structure:
```
/Users/satvikpraveen/Desktop/ExperimentP5/    ← Main workspace
├── configs/
├── data/
├── mimagen_pytorch/
├── transformer/                               ← Model definitions
│   ├── sequentialModel.py
│   └── spatialModel.py
├── ExperimentP5/                              ⚠️ Redundant folder
│   └── transformer/                           ← Empty folder
└── ...
```

### Issue Identified

**The `ExperimentP5/transformer/` folder is empty and redundant.**

This appears to be leftover from a previous repository structure or Git clone operation. The actual transformer models are located at:
- **Correct location**: `/Users/satvikpraveen/Desktop/ExperimentP5/transformer/`
- **Contains**: `sequentialModel.py` and `spatialModel.py`

The nested `ExperimentP5/` directory serves no purpose and should be removed.

### Recommendation

**Remove the redundant folder**:
```bash
# From workspace root
rm -rf ExperimentP5/
```

This will:
- ✅ Clean up the repository structure
- ✅ Avoid confusion about which folder to use
- ✅ Prevent potential import issues
- ✅ Make the repository cleaner for cluster deployment

### Correct Repository Structure (After Cleanup)

```
ExperimentP5/                           ← Workspace root
├── configs/
│   └── base_config.py                  ← Configurations with split definitions
├── data/
│   └── optimized_data.py               ← Dataset classes with split support
├── transformer/                        ← Model definitions (GPT-2, Spatial)
│   ├── sequentialModel.py
│   └── spatialModel.py
├── train_test_seq/
│   └── optimized_trainer.py            ← Sequential trainer with validation
├── train_test_spatial/
│   └── optimized_trainer.py            ← Diffusion trainer
├── mimagen_pytorch/                    ← Imagen implementation
├── scripts/                            ← SLURM submission scripts
├── train_sequential.py                 ← Sequential training entry point
├── train_diffusion.py                  ← Diffusion training entry point
├── eval_sequential.py                  ← Sequential evaluation with split selection
├── eval_diffusion.py                   ← Diffusion evaluation
├── output/                             ← Training outputs (models, logs)
├── venv/                               ← Virtual environment
└── requirements.txt
```

---

## Summary

### ✅ Data Splits: Fully Implemented

1. **Three splits defined**: Train (8000), Validation (500), Test (512)
2. **Used in training**: Validation performed during training loops
3. **Used in evaluation**: Can evaluate on any split (train/valid/test)
4. **Configurable**: Each preset can override split parameters
5. **Non-overlapping**: Sequential timestep ranges ensure no data leakage
6. **Both modes**: Works for sequential (BFS) and diffusion (DICOM) datasets

### ⚠️ Repository Structure: Needs Cleanup

1. **Remove**: `ExperimentP5/` nested folder (empty, redundant)
2. **Use**: Top-level `transformer/` folder for model definitions
3. **Impact**: No code changes needed, just folder deletion

### Verification Command

To see the actual split ranges being used:
```bash
python train_sequential.py --config optimized_default

# Output shows:
# 📊 Data Configuration:
#   Train: 0 → 8000
#   Valid: 8000 → 8500
#   Image Size: 720×432
```

---

## Next Steps

1. **Clean up structure**:
   ```bash
   rm -rf ExperimentP5/
   ```

2. **Verify splits work**:
   ```bash
   python -c "
   from configs.base_config import get_experiment_config
   config = get_experiment_config('optimized_default')
   print(f'Train: {config.data.train_start}-{config.data.train_start + config.data.train_span}')
   print(f'Valid: {config.data.valid_start}-{config.data.valid_start + config.data.valid_span}')
   print(f'Test: {config.data.test_start}-{config.data.test_start + config.data.test_span}')
   "
   ```

3. **Test with debug mode** (uses all splits):
   ```bash
   ./scripts/submit_job.sh debug
   ```
