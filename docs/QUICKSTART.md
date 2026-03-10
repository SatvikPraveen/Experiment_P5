# ExperimentP5 Quick Start Guide

## 🎯 Get Started in 5 Minutes

### 1. Environment Setup (2 min)

```bash
# Clone/navigate to ExperimentP5
cd ExperimentP5

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### 2. Quick Test Run (1 min)

```bash
# Debug mode: 2 epochs only to verify everything works
python train_sequential.py --debug

# Expected output:
# ✅ Configuration loaded
# ✅ Data loading working
# ✅ Model created
# ✅ Training starts
# ⏱️  Takes ~2-5 minutes depending on GPU
```

### 3. Full Training (local machine)

```bash
# Sequential model (GPU recommended)
python train_sequential.py \
    --preset optimized_default \
    --epochs 100 \
    --batch_size 16 \
    --amp \
    --multi_gpu

# Diffusion model
python train_diffusion.py \
    --preset optimized_default \
    --epochs 100 \
    --batch_size 8 \
    --amp \
    --multi_gpu
```

### 4. Cluster Training with SLURM

```bash
# Make submission script executable
chmod +x scripts/submit_job.sh

# Submit sequential training
./scripts/submit_job.sh seq PRESET=optimized_default EPOCHS=100

# Submit diffusion training
./scripts/submit_job.sh diff PRESET=720_816_medium_unet EPOCHS=100

# Full pipeline (seq → diff automatically)
./scripts/submit_job.sh pipeline

# Monitor job
squeue -u $USER
tail -f logs/seq_*.out
```

---

## 📊 Available Configurations

Quick reference for presets:

| Preset | Use Case | Speed | Quality | Memory |
|--------|----------|-------|---------|--------|
| `small_fast` | Prototyping | ⚡⚡⚡ | ⭐⭐ | 🟢 Low |
| `optimized_default` | **Recommended** | ⚡⚡ | ⭐⭐⭐ | 🟡 Medium |
| `720_816_medium_unet` | Balance | ⚡⚡ | ⭐⭐⭐⭐ | 🟡 Medium |
| `720_1024_large_unet` | Best quality | ⚡ | ⭐⭐⭐⭐⭐ | 🔴 High |
| `high_resolution` | Publication | ⚡ | ⭐⭐⭐⭐⭐ | 🔴 High |

---

## 🔧 Common Commands

### Training

```bash
# Resume from checkpoint
python train_sequential.py --resume output/seq_model/checkpoints/epoch_50.pt

# Override specific parameters
python train_sequential.py --preset optimized_default --lr 5e-5 --batch_size 32

# Specify GPU
CUDA_VISIBLE_DEVICES=1 python train_sequential.py --preset optimized_default
```

### Evaluation

```bash
# Evaluate sequential model
python eval_sequential.py \
    --checkpoint output/seq_model/checkpoints/best.pt \
    --split valid \
    --save_predictions

# Evaluate diffusion model
python eval_diffusion.py \
    --checkpoint output/diff_model/checkpoints/ \
    --num_samples 32 \
    --save_samples
```

### Monitoring

```bash
# Watch training progress
tail -f output/seq_model/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# View metrics (Python)
python -c "
import json
with open('output/seq_model/metrics/metrics.json') as f:
    m = json.load(f)
    print(f'Latest loss: {m["train_loss"][-1]:.6f}')
"
```

---

## 📁 Output Structure

After training, you'll have:

```
output/
├── <experiment_name>/
│   ├── config.json                    # Saved configuration
│   ├── training.log                   # Full training log
│   ├── checkpoints/
│   │   ├── best.pt                    # Best model (lowest val loss)
│   │   ├── latest.pt                  # Latest checkpoint
│   │   ├── epoch_10.pt                # Periodic checkpoints
│   │   └── epoch_20.pt
│   ├── metrics/
│   │   ├── metrics.json               # Training/validation curves
│   │   └── gpu_stats.json             # GPU utilization
│   ├── plots/
│   │   ├── loss_curve.png             # Loss visualization
│   │   └── lr_schedule.png            # Learning rate plot
│   └── eval_results/                  # (after evaluation)
│       ├── predictions_valid.npy
│       ├── targets_valid.npy
│       └── predictions_viz_valid.png
```

---

## ⚡ Performance Tips

### Getting 4-6x Speedup

1. **Enable AMP** (1.5x faster, less memory)
   ```bash
   --amp
   ```

2. **Use Multi-GPU** (1.8x with 2 GPUs)
   ```bash
   --multi_gpu
   ```

3. **Increase Workers** (3-4x faster data loading)
   ```bash
   --num_workers 8
   ```

4. **Larger Batches** (better GPU utilization)
   ```bash
   --batch_size 32  # adjust based on GPU memory
   ```

5. **Gradientccumulation** (if OOM with large batches)
   ```python
   # Edit config:
   config.training.gradient_accumulation_steps = 4
   ```

### Memory Management

If you get **CUDA Out of Memory**:

```bash
# Reduce batch size
python train_sequential.py --batch_size 8

# Or use gradient accumulation (edit config)
# Set gradient_accumulation_steps = 2 or 4

# For diffusion model, enable CPU offloading
# (Edit trainer to set cpu_offload=True in ImagenTrainer)
```

---

## 🐛 Troubleshooting

### Issue: Import errors

**Solution:** Make sure you're in the ExperimentP5 directory
```bash
cd /path/to/ExperimentP5
python train_sequential.py --debug
```

### Issue: Data not found

**Solution:** Check data paths in config
```python
# In configs/base_config.py, update paths:
data_dir = '/path/to/your/data'
```

### Issue: ASTRA not installed (for DICOM/sinograms)

**Solution:** Install via conda
```bash
conda install -c astra-toolbox astra-toolbox
```

Or use BFS data only (no DICOM):
```python
config.data.data_type = 'bfs'  # Not 'dicom'
```

### Issue: Multi-GPU not working

**Check:**
```bash
# Verify GPUs visible
nvidia-smi

# Verify CUDA
python -c "import torch; print(torch.cuda.device_count())"

# Specify GPUs
CUDA_VISIBLE_DEVICES=0,1 python train_sequential.py --multi_gpu
```

### Issue: Training is slow

**Checklist:**
- [ ] Using GPU? (`--device cuda`)
- [ ] AMP enabled? (`--amp`)
- [ ] Enough workers? (`--num_workers 8`)
- [ ] Batch size > 1? (`--batch_size 16`)
- [ ] Check `nvidia-smi` - GPU utilization should be >80%

---

## 📈 Next Steps

### After Training

1. **Evaluate model**
   ```bash
   python eval_sequential.py --checkpoint output/seq_model/checkpoints/best.pt
   ```

2. **Visualize results**
   - Check `output/plots/` for training curves
   - Check `output/eval_results/` for prediction visualizations

3. **Fine-tune**
   - Adjust learning rate if loss plateaus early
   - Try different presets for quality/speed tradeoff
   - Enable curriculum learning (already in sequential trainer)

4. **Scale up**
   - Train diffusion model with sequential model conditioning
   - Increase resolution for final results
   - Run full pipeline end-to-end

### Production Deployment

1. **Export model**
   ```python
   # Convert to TorchScript
   model_scripted = torch.jit.trace(model, example_input)
   model_scripted.save('model.pt')
   ```

2. **Optimize inference**
   - Use `torch.compile()` (PyTorch 2.0+)
   - Enable FP16 inference
   - Batch predictions

3. **Monitor in production**
   - Log predictions
   - Track inference time
   - Monitor GPU memory

---

## 📚 More Resources

- **Full Documentation**: [README.md](README.md)
- **Configuration Details**: [configs/base_config.py](configs/base_config.py)
- **Optimization Guide**: `../TRAINING_OPTIMIZATIONS.md` (in parent directory)
- **Original Paper**: Nature Communications (2024) - G-LED

---

## ✅ Checklist for First Run

- [ ] Python 3.9+ installed
- [ ] CUDA and PyTorch working (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data paths configured (check `configs/base_config.py`)
- [ ] Quick test passed (`python train_sequential.py --debug`)
- [ ] Ready for full training!

---

**Need help?** Check the full [README.md](README.md) or refer to the original repository documentation.

**Happy Training! 🚀**
