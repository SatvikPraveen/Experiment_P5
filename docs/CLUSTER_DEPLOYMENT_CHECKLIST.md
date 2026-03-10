# Cluster Deployment Checklist ✅

**Status: READY FOR DEPLOYMENT**
All tests passed on: 2026-03-09

---

## Pre-Deployment Verification

✅ **All imports working** - No missing dependencies  
✅ **File structure complete** - All required files present  
✅ **SLURM scripts validated** - Correct paths, GPU specs, PROJECT_DIR set  
✅ **Configuration tested** - Config loading works properly  
✅ **Data pipeline ready** - Both BFS and DICOM pipelines functional  
✅ **No interactive inputs** - Training scripts won't block on cluster  

---

## Deployment Steps

### 1. Transfer Files to Cluster

```bash
# From your local machine:
rsync -av --progress ~/Desktop/Experiments_P5/ \
  satvikpraveen@forest.usf.edu:/home/s/satvikpraveen/Experiments_P5/

# Exclude unnecessary files (if needed):
rsync -av --progress \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  ~/Desktop/Experiments_P5/ \
  satvikpraveen@forest.usf.edu:/home/s/satvikpraveen/Experiments_P5/
```

### 2. Set Up Cluster Environment

```bash
# SSH to cluster
ssh satvikpraveen@forest.usf.edu

# Navigate to project
cd /home/s/satvikpraveen/Experiments_P5

# Activate conda environment
conda activate sam_gpu

# Install DICOM dependencies (if using DICOM pipeline)
pip install SimpleITK astra-toolbox

# Verify installation
python -c "import SimpleITK; import astra; print('✅ DICOM tools ready')"
```

### 3. Verify Data Paths

**For BFS Dataset:**
- Ensure data files are at: `/home/s/satvikpraveen/Experiments_P5/data/data0.npy`, `data1.npy`
- Check [configs/base_config.py](configs/base_config.py) data_location paths

**For DICOM Dataset:**
- Ensure DICOM files are at: `/home/s/satvikpraveen/Experiments_P5/data/Dataset/`
- Verify [examples/dicom_fbp_config.py](examples/dicom_fbp_config.py) dicom_path

### 4. Test Before Submitting Jobs

```bash
# Quick import test
python -c "from mimagen_pytorch.physics_diffusion import Imagen; print('✅ Models import OK')"

# Config test
python -c "from configs.base_config import get_preset_config; c=get_preset_config('optimized_default'); print('✅ Config OK')"

# Data loading test (for BFS)
python -c "from data.optimized_data import create_dataloaders_from_config; print('✅ Data pipeline OK')"
```

### 5. Submit Your First Job

**For Diffusion Model (single job):**
```bash
sbatch scripts/train_diff.slurm
```

**For Sequential Model (single job):**
```bash
sbatch scripts/train_seq.slurm
```

**For Chained Jobs (>23 hours):**
```bash
# Diffusion with auto-restart
sbatch scripts/train_diff_chain.slurm

# Sequential with auto-restart
sbatch scripts/train_seq_chain.slurm
```

### 6. Monitor Your Jobs

```bash
# Check job status
squeue -u satvikpraveen

# View live logs
tail -f logs/diff_<JOBID>.out

# Check for errors
tail -f logs/diff_<JOBID>.err

# Cancel a job if needed
scancel <JOBID>
```

---

## SLURM Job Configuration

All scripts configured with:
- **Partition:** Quick (23-hour limit)
- **GPUs:** 2x A100, A40, or TitanRTX (NOT TitanX - incompatible)
- **CPUs:** 8 per task
- **Memory:** 64GB
- **Environment:** sam_gpu conda environment
- **Paths:** Updated to Experiments_P5

---

## Troubleshooting

### If job fails immediately:
```bash
# Check error log
cat logs/diff_<JOBID>.err

# Verify conda environment
conda activate sam_gpu
python --version  # Should be Python 3.10+
```

### If GPU not available:
```bash
# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"

# Modify SLURM script to request different GPU:
# Change: #SBATCH --gres=gpu:A100:2
# To:     #SBATCH --gres=gpu:A40:2
```

### If data not found:
```bash
# Verify data paths
ls -lh data/*.npy          # For BFS
ls -lh data/Dataset/*.dcm  # For DICOM

# Check config matches actual paths
grep -n "data_location\|dicom_path" configs/base_config.py
```

### Memory issues:
- BFS dataset uses memory-mapped loading (already optimized)
- DICOM uses caching to avoid reprocessing
- If still issues, reduce batch_size in config

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `train_diffusion.py` | Main diffusion training script |
| `train_sequential.py` | Main sequential training script |
| `configs/base_config.py` | Central configuration |
| `examples/dicom_fbp_config.py` | DICOM example config |
| `scripts/train_diff.slurm` | Diffusion SLURM job |
| `scripts/train_seq.slurm` | Sequential SLURM job |
| `scripts/train_diff_chain.slurm` | Auto-chaining diffusion job |
| `scripts/train_seq_chain.slurm` | Auto-chaining sequential job |

---

## Expected Outputs

Jobs will create:
- **Logs:** `logs/diff_<JOBID>.out` and `.err`
- **Checkpoints:** `checkpoints/*.pt`
- **Results:** `output/`
- **WandB:** Logs sent to wandb.ai (if configured)

---

## Post-Deployment

After successful job completion:
1. **Download results:**
   ```bash
   rsync -av satvikpraveen@forest.usf.edu:/home/s/satvikpraveen/Experiments_P5/output/ ~/Desktop/Experiments_P5/output/
   ```

2. **Download checkpoints:**
   ```bash
   rsync -av satvikpraveen@forest.usf.edu:/home/s/satvikpraveen/Experiments_P5/checkpoints/ ~/Desktop/Experiments_P5/checkpoints/
   ```

3. **Run evaluation:**
   ```bash
   python eval_diffusion.py --checkpoint checkpoints/best_model.pt
   ```

---

## Additional Notes

- **Auto-chaining:** Chain scripts automatically submit next job before timeout
- **Memory-mapped data:** BFS dataset loads efficiently without RAM overflow
- **DICOM caching:** First run processes all DICOM files, subsequent runs use cache
- **GPU compatibility:** Scripts avoid old TitanX GPUs (CUDA 5.2 not supported)

**Need help?** Check [docs/](docs/) folder for detailed documentation.
