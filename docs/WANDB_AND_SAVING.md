# 📊 Wandb Monitoring & Model Saving Guide

## 🎯 Overview

ExperimentP5 now includes comprehensive experiment tracking with **Weights & Biases (wandb)** and automatic model checkpointing.

---

## 🔧 Setup Wandb

### 1. Install Wandb (Already in requirements.txt)

```bash
source venv/bin/activate
pip install wandb
```

### 2. Login to Wandb

```bash
wandb login
```

Enter your API key when prompted. Get your key from: https://wandb.ai/authorize

### 3. (Optional) Set Your Entity

Edit `configs/base_config.py`:
```python
wandb_entity: Optional[str] = "your-username"  # Your wandb username/team
```

---

## 📋 Unique Project Names Per Configuration

Each configuration automatically gets a **unique wandb project** based on:
- **Base project**: `experimentp5-gled`
- **Mode**: `seq` (sequential) or `diff` (diffusion)
- **Preset name**: e.g., `optimized_default`, `720_816_medium_unet`
- **Run name**: `seed42` (based on random seed)

### Project Naming Pattern

```
experimentp5-gled-{mode}-{preset_name}
```

### Examples:

| Configuration | Wandb Project | Run Name |
|--------------|---------------|----------|
| Sequential, optimized_default, seed=42 | `experimentp5-gled-seq-optimized_default` | `seed42` |
| Sequential, 720_816_medium_unet, seed=123 | `experimentp5-gled-seq-720_816_medium_unet` | `seed123` |
| Diffusion, optimized_default, seed=42 | `experimentp5-gled-diff-optimized_default` | `seed42` |
| Diffusion, updated_1024, seed=456 | `experimentp5-gled-diff-updated_1024_512` | `seed456` |

### Reproducibility Study Projects

When running multiple seeds for the same preset:
- All runs for the same preset go to the **same project**
- Different seeds create different **runs** within that project
- Easy comparison between seeds!

Example - Three seeds for one preset:
```
Project: experimentp5-gled-seq-optimized_default
├── Run: seed42
├── Run: seed123
└── Run: seed456
```

---

## 📊 What Gets Logged to Wandb

### Configuration (logged at start)
- Preset name
- Mode (sequential/diffusion)
- Batch size
- Learning rate
- Number of epochs
- Random seed
- Image dimensions
- Model architecture details

### Metrics (logged every epoch)
- **train_loss**: Training loss
- **valid_loss**: Validation loss (when evaluated)
- **learning_rate**: Current learning rate
- **best_loss**: Best loss so far (diffusion only)
- **Nt**: Curriculum learning progress (sequential only)

### Tags
Each run is tagged for easy filtering:
- Mode: `sequential` or `diffusion`
- Preset name: e.g., `optimized_default`
- Seed: e.g., `seed42`

---

## 🚀 Using Wandb

### Enable Wandb (Default: ON)

```bash
# Wandb is enabled by default
python train_sequential.py --preset optimized_default

# Explicitly enable
python train_sequential.py --preset optimized_default --wandb
```

### Disable Wandb

```bash
# Disable wandb logging
python train_sequential.py --preset optimized_default --no-wandb
```

### Custom Wandb Settings

```bash
# Custom project name
python train_sequential.py \
    --preset optimized_default \
    --wandb_project "my-custom-project" \
    --wandb_entity "my-team"

# Multiple seeds with wandb
python train_sequential.py --preset optimized_default --seed 42 --wandb
python train_sequential.py --preset optimized_default --seed 123 --wandb
python train_sequential.py --preset optimized_default --seed 456 --wandb
```

### SLURM with Wandb

All SLURM scripts automatically use wandb (if enabled in config):

```bash
./scripts/submit_job.sh seq PRESET=optimized_default SEED=42
./scripts/submit_job.sh diff PRESET=720_816_medium_unet SEED=123
```

---

## 💾 Model Saving

### Automatic Checkpointing

Models are automatically saved during training:

#### Sequential Model Saves:
```
output/<experiment_name>_<timestamp>/checkpoints/
├── checkpoint_epoch_0001.pt    # Every N epochs (configurable)
├── checkpoint_epoch_0002.pt
├── ...
├── best_model.pt               # Best model based on validation loss
└── final_model.pt              # Model after final epoch
```

#### Diffusion Model Saves:
```
output/<experiment_name>_<timestamp>/checkpoints/
├── best_model_sofar            # Best model (ImagenTrainer format)
├── best_model_sofar_epoch      # Epoch number for best model
└── final_model                 # Final model (ImagenTrainer format)
```

### Save Locations

After training completes, check the output messages:

```
✅ Training script complete!

📁 Model saved to: output/optimized_default_20260306_143022/checkpoints
   - Best model: output/optimized_default_20260306_143022/checkpoints/best_model.pt
   - Final model: output/optimized_default_20260306_143022/checkpoints/final_model.pt
   - Checkpoints: output/optimized_default_20260306_143022/checkpoints/checkpoint_epoch_*.pt
```

### What's Saved in Each Checkpoint

#### Sequential (.pt files):
```python
{
    'epoch': 50,
    'model_state_dict': {...},          # Model weights
    'optimizer_state_dict': {...},      # Optimizer state
    'scheduler_state_dict': {...},      # LR scheduler state
    'metrics': {'loss': 0.045, ...},    # Performance metrics
    'timestamp': '2026-03-06T14:30:22'  # Save time
}
```

#### Diffusion (ImagenTrainer format):
Uses the ImagenTrainer's built-in save/load mechanism.

### Loading Saved Models

#### Sequential Model:
```python
from configs.base_config import ExperimentConfig
from transformer.sequentialModel import SequentialModel

# Load config
config = ExperimentConfig.load('output/experiment/logs/config.json')

# Create model
model = SequentialModel(config)

# Load checkpoint
checkpoint = torch.load('output/experiment/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Also get epoch, optimizer state, etc. from checkpoint
epoch = checkpoint['epoch']
metrics = checkpoint['metrics']
```

#### Diffusion Model:
```python
from train_test_spatial.optimized_trainer import OptimizedDiffusionTrainer

# Create trainer (with config and model)
trainer = OptimizedDiffusionTrainer(...)

# Load best model
trainer.trainer.load('output/experiment/checkpoints/best_model_sofar')
```

---

## 🔍 Monitoring Training

### Wandb Dashboard

1. **Go to wandb.ai** and navigate to your project
2. **View real-time metrics**: Loss curves, learning rate, etc.
3. **Compare runs**: Multiple seeds, different presets
4. **Filter by tags**: Show only sequential, only seed42, etc.

### Example Wandb Views

**Grouped by Preset:**
- Compare different presets side-by-side
- See which configuration performs best

**Grouped by Seed:**
- Compare reproducibility across seeds
- Verify that same seed produces same results

**Grouped by Mode:**
- Compare sequential vs diffusion
- Analyze two-stage pipeline

### Local Logs

Besides wandb, logs are also saved locally:

```
output/<experiment_name>/logs/
├── config.json              # Experiment configuration
├── metrics.json             # Training metrics history
├── training.log             # Detailed training log
└── plots/                   # Metric plots (if generated)
    ├── loss_curve.png
    └── lr_schedule.png
```

---

## 📈 Reproducibility Study with Wandb

### Running Multiple Seeds

```bash
# Option 1: Manual
python train_sequential.py --preset optimized_default --seed 42 --wandb
python train_sequential.py --preset optimized_default --seed 123 --wandb
python train_sequential.py --preset optimized_default --seed 456 --wandb

# Option 2: Use reproducibility script
./run_reproducibility_study.sh
```

### Analyzing in Wandb

All runs with the same preset will be in the **same project**, making comparison easy:

1. Go to project (e.g., `experimentp5-gled-seq-optimized_default`)
2. See all runs (seed42, seed123, seed456)
3. Compare metrics:
   - Overlay loss curves
   - Check variance across seeds
   - Verify reproducibility

### Expected Variance

With `deterministic=True` and same seed:
- **Exact same results** ✅
- Loss values identical
- Model weights identical

With different seeds:
- **Similar but not identical** ✓
- Final loss should be within small range
- Trends should be consistent

---

## ⚙️ Configuration Options

### In `configs/base_config.py`:

```python
@dataclass
class TrainingConfig:
    # Wandb settings
    use_wandb: bool = True                          # Enable/disable
    wandb_project: str = "experimentp5-gled"       # Base project name
    wandb_entity: Optional[str] = None              # Your username/team
    
    # Checkpointing
    save_every: int = 1                             # Save checkpoint every N epochs
    keep_last_n_checkpoints: int = 5                # Keep only last N checkpoints
```

### Command-line overrides:

```bash
# Disable wandb for this run
--no-wandb

# Custom wandb project
--wandb_project "my-experiment"

# Custom wandb entity (team)
--wandb_entity "my-team"

# Change checkpoint frequency
--save_every 10  # (not yet implemented, but can be added)
```

---

## 🎯 Best Practices

### 1. **Always Use Wandb for Important Experiments**
- Free for academics
- Invaluable for comparing configurations
- Easy to share results

### 2. **Use Meaningful Seeds**
- Common: 42, 123, 456 for reproducibility studies
- Document which seed produced best results

### 3. **Tag Your Runs**
- Automatically tagged with preset, mode, seed
- Add custom tags if needed (code modification)

### 4. **Check Both Best and Final Models**
- Best model: Lowest validation loss
- Final model: After all epochs (may overfit)

### 5. **Monitor During Training**
- Check wandb dashboard periodically
- Catch issues early (NaN loss, etc.)

### 6. **Archive Important Models**
```bash
# Copy best models to safe location
cp output/experiment/checkpoints/best_model.pt ~/models/seq_optimized_seed42.pt
```

---

## 📋 Quick Reference

### Training Commands

| Task | Command |
|------|---------|
| Train with wandb | `python train_sequential.py --preset optimized_default` |
| Train without wandb | `python train_sequential.py --preset optimized_default --no-wandb` |
| Custom seed | `python train_sequential.py --preset optimized_default --seed 123` |
| Custom wandb project | `python train_sequential.py --wandb_project "my-project"` |

### Model Locations

| Model Type | Location Pattern |
|-----------|------------------|
| Sequential best | `output/*/checkpoints/best_model.pt` |
| Sequential final | `output/*/checkpoints/final_model.pt` |
| Diffusion best | `output/*/checkpoints/best_model_sofar` |
| Diffusion final | `output/*/checkpoints/final_model` |

### Wandb Projects

| Configuration | Project Name |
|--------------|--------------|
| Sequential + preset | `experimentp5-gled-seq-<preset>` |
| Diffusion + preset | `experimentp5-gled-diff-<preset>` |

---

## 🐛 Troubleshooting

### "wandb not available"
```bash
pip install wandb
wandb login
```

### "Permission denied" on wandb
```bash
# Login with your credentials
wandb login

# Or disable wandb
python train_sequential.py --no-wandb
```

### Can't find saved models
```bash
# Check output directory
ls -R output/

# Models are in: output/<experiment_name>_<timestamp>/checkpoints/
```

### Wandb not logging
- Check that `use_wandb=True` in config
- Check that wandb is installed: `pip show wandb`
- Check internet connection
- Try `wandb login` again

---

## ✅ Summary

✅ **Wandb enabled** by default for experiment tracking  
✅ **Unique project per configuration** for organized monitoring  
✅ **Automatic model checkpointing** with best model tracking  
✅ **Multiple save formats** (.pt for sequential, ImagenTrainer for diffusion)  
✅ **Complete metrics logging** (loss, LR, custom metrics)  
✅ **Easy reproducibility** with seed-based run names  
✅ **Local + cloud storage** (both filesystem and wandb)  

Your experiments are now fully tracked and all models are automatically saved! 🎉
