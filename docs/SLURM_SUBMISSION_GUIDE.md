# SLURM Submission Guide

## Overview

The `scripts/` directory contains multiple SLURM job files, each optimized for different training scenarios. This modular approach provides flexibility in resource allocation and workflow management.

## Why Multiple SLURM Files?

**Resource Efficiency**: Different training modes have different computational requirements:
- Sequential model training is faster (48 hours)
- Diffusion model training requires more time (72 hours)
- Full pipeline needs extended time (120 hours)
- Debug mode minimizes resource usage (2 hours, 1 GPU)

**Workflow Flexibility**: Run specific parts of the pipeline independently without wasting cluster resources.

**Easier Maintenance**: Separate files are clearer and easier to modify than one monolithic file with complex conditionals.

**Cluster Efficiency**: Request only the resources you need, improving job queue priority and cluster utilization.

## SLURM Files Reference

### NEW: Auto-Chaining Scripts for Full Convergence 🔄

For training beyond the 23-hour Quick partition limit, use the auto-chaining scripts that automatically resubmit with checkpointing:

#### `train_seq_chain.slurm`
**Purpose**: Sequential training with automatic job chaining for full convergence

**How it works**:
1. Trains for up to 23 hours
2. Saves checkpoint when time limit approaches
3. Automatically resubmits itself with `--resume`
4. Continues until TARGET_EPOCHS reached (default: 500)

**Usage**:
```bash
sbatch --export=PRESET=optimized_default,TARGET_EPOCHS=500,SEED=42 scripts/train_seq_chain.slurm
```

#### `train_diff_chain.slurm`
**Purpose**: Diffusion training with automatic job chaining

**How it works**: Same as sequential but for diffusion model

**Usage**:
```bash
sbatch --export=PRESET=optimized_default,TARGET_EPOCHS=500,SEED=42 scripts/train_diff_chain.slurm
```

#### `run_all_configs.sh`
**Purpose**: Submit all 7 preset configurations for complete reproducibility study

**Usage**:
```bash
# Train all presets (both sequential and diffusion)
./scripts/run_all_configs.sh both

# Only sequential models for all presets
./scripts/run_all_configs.sh seq

# Only diffusion models for all presets
./scripts/run_all_configs.sh diff
```

**Features**:
- Submits all 7 configurations automatically
- Diffusion jobs wait for their sequential model to finish
- Each job auto-chains every 23 hours
- Tracks all jobs with unique wandb projects

---

### 1. `debug.slurm`
**Purpose**: Quick testing and validation before full deployment

**Resources**:
- Time: 2 hours
- GPUs: 1
- Memory: 32GB

**Training Configuration**:
- Trains both sequential and diffusion models
- Reduced epochs: 2 for sequential, 2 for diffusion
- Quick validation of entire pipeline

**When to use**:
- Testing new configurations
- Validating code changes
- Verifying data loading
- Pre-deployment sanity checks

**Usage**:
```bash
./scripts/submit_job.sh debug
# Or with custom preset:
./scripts/submit_job.sh debug PRESET=720_816_standard SEED=42
```

---

### 2. `train_seq.slurm`
**Purpose**: Train only the sequential transformer model

**Resources**:
- Time: 48 hours
- GPUs: 2
- Memory: 64GB

**Training Configuration**:
- Sequential model only (GPT-2 style transformer)
- batch_size: 16
- Full training epochs based on config

**When to use**:
- Training sequential model independently
- Faster turnaround than full pipeline
- Sequential model development/tuning
- When you already have a trained diffusion model

**Usage**:
```bash
./scripts/submit_job.sh seq PRESET=optimized_default SEED=42
```

---

### 3. `train_diff.slurm`
**Purpose**: Train only the diffusion model (Imagen)

**Resources**:
- Time: 72 hours
- GPUs: 2
- Memory: 64GB

**Training Configuration**:
- Diffusion model only (Imagen-based)
- batch_size: 8 (diffusion is more memory-intensive)
- Full training epochs based on config

**When to use**:
- Training diffusion model independently
- Diffusion model development/tuning
- When you already have a trained sequential model
- Testing different U-Net architectures

**Usage**:
```bash
./scripts/submit_job.sh diff PRESET=720_816_high_unet SEED=123
```

---

### 4. `full_pipeline.slurm`
**Purpose**: Train both sequential and diffusion models sequentially

**Resources**:
- Time: 120 hours (5 days)
- GPUs: 2
- Memory: 64GB

**Training Configuration**:
- Trains sequential model first
- Then trains diffusion model
- Complete end-to-end pipeline for one configuration

**When to use**:
- Complete experiment from scratch
- Production training runs
- Comparative studies across presets
- When you need both models trained consistently

**Usage**:
```bash
./scripts/submit_job.sh pipeline PRESET=updated_1024 SEED=42
```

---

## Submit Job Helper Script

### `submit_job.sh`
**Purpose**: Convenience wrapper for submitting SLURM jobs

**Features**:
- Validates mode selection
- Sets up environment variables
- Provides consistent parameter passing
- Simplifies job submission syntax

**Usage Pattern**:
```bash
./scripts/submit_job.sh <MODE> [PRESET=<name>] [SEED=<value>]
```

**Modes**:
- `debug`: Quick 2-hour test
- `seq`: Sequential model training (48h)
- `diff`: Diffusion model training (72h)
- `pipeline`: Full pipeline (120h)

**Examples**:
```bash
# Test with debug mode
./scripts/submit_job.sh debug

# Train sequential model with custom preset and seed
./scripts/submit_job.sh seq PRESET=720_816_standard SEED=42

# Train diffusion model
./scripts/submit_job.sh diff PRESET=720_816_high_unet SEED=123

# Full pipeline with specific configuration
./scripts/submit_job.sh pipeline PRESET=updated_1024 SEED=2024
```

---

## Common SLURM Features (All Files)

All SLURM scripts include:

**Robust Error Handling**:
```bash
#!/bin/bash -l
set -euo pipefail
```
- `-e`: Exit on any error
- `-u`: Exit on undefined variables
- `-o pipefail`: Catch errors in pipes

**Environment Setup**:
- Module loading (commented out, uncomment for your cluster)
- Virtual environment activation
- Working directory setup

**Logging**:
- Timestamped output files: `logs/slurm-%j.out`
- Detailed progress messages
- Training completion notifications

**Configurable Parameters**:
- `PRESET`: Configuration to use (default: optimized_default)
- `SEED`: Random seed (default: 42)

---

## Recommended Workflow

### First Time Setup
1. **Test locally** (if possible):
   ```bash
   python train_sequential.py --config optimized_default
   ```

2. **Debug on cluster**:
   ```bash
   ./scripts/submit_job.sh debug
   ```
   Check logs to verify everything works

3. **Run production training**:
   ```bash
   ./scripts/submit_job.sh pipeline PRESET=optimized_default SEED=42
   ```

### Development Workflow
1. Make code changes
2. Test with `debug` mode
3. Once validated, run appropriate mode (`seq`, `diff`, or `pipeline`)

### Production Experiments
For reproducibility studies across all presets:
```bash
# Run each preset with consistent seed
for preset in baseline 720_816_standard 720_816_high_unet 720_816_medium_unet \
              720_1024_large_unet updated_1024 feather_720_432 optimized_default; do
    ./scripts/submit_job.sh pipeline PRESET=$preset SEED=42
done
```

---

## Resource Requirements Summary

| File | Time | GPUs | Memory | Use Case |
|------|------|------|--------|----------|
| `debug.slurm` | 2h | 1 | 32GB | Quick testing |
| `train_seq.slurm` | 48h | 2 | 64GB | Sequential only |
| `train_diff.slurm` | 72h | 2 | 64GB | Diffusion only |
| `full_pipeline.slurm` | 120h | 2 | 64GB | Complete pipeline |

---

## Output Organization

All training runs create timestamped directories:
```
output/
├── {preset_name}_{timestamp}/
│   ├── sequential/
│   │   ├── model_epoch_*.pt
│   │   └── final_model.pt
│   └── diffusion/
│       ├── unet_epoch_*.pt
│       └── final_unet.pt
└── ...
```

Logs are saved to:
```
logs/
└── slurm-{job_id}.out
```

---

## Monitoring

All jobs integrate with Weights & Biases (wandb):
- **Project naming**: `experimentp5-gled-{mode}-{preset_name}`
- **Run naming**: Includes timestamp and configuration details
- **Metrics tracked**: Training loss, validation metrics, learning rate, epoch progress

Access your runs at: https://wandb.ai/

---

## Troubleshooting

**Job fails immediately**:
- Check module loads are uncommented and correct for your cluster
- Verify virtual environment exists: `venv/`
- Check data paths in `configs/base_config.py`

**Out of memory errors**:
- Reduce batch size in the SLURM file
- Use debug mode first to test memory requirements
- Consider using a preset with smaller dimensions

**Job times out**:
- Increase time limit in SLURM file
- Reduce number of epochs in config
- Check if validation is taking excessive time

**Data loading errors**:
- Verify `data_root` path in `configs/base_config.py`
- Ensure data is accessible from compute nodes
- Check file permissions

---

## Before Cluster Submission

Complete this checklist:

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Configure data paths in `configs/base_config.py`
- [ ] Uncomment and configure module loads in SLURM scripts
- [ ] Login to wandb: `wandb login`
- [ ] Test with debug mode: `./scripts/submit_job.sh debug`
- [ ] Review output and logs for any issues
- [ ] Submit production job with appropriate mode

See `CLUSTER_CHECKLIST.md` for complete deployment checklist.
