# Long-Running Jobs (>5 Days)

This folder contains SLURM scripts that require more than 5 days (120 hours) of continuous runtime.

## Files

### `full_pipeline.slurm`
**Purpose**: Train both sequential and diffusion models sequentially in a single job

**Time Required**: ~7-10 days with 500 epochs for both models

**When to Use**: 
- Only if your cluster has a partition that allows >5 days
- For end-to-end pipeline without job dependencies

**Note**: This won't work with the Quick partition (23-hour limit).

## Recommended Alternative

For clusters with time limits, use the auto-chaining scripts in the parent directory instead:

```bash
# This handles the full pipeline with auto-resubmission
./scripts/run_all_configs.sh both
```

The chaining scripts (`train_seq_chain.slurm`, `train_diff_chain.slurm`) accomplish the same result by:
1. Running for 23 hours
2. Saving checkpoint
3. Auto-resubmitting continuation job
4. Repeating until 500 epochs complete

## Usage (If You Have Long Partition)

If your cluster has a `long`, `batch`, or similar partition allowing >5 days:

1. Edit `full_pipeline.slurm` and change partition:
   ```bash
   #SBATCH -p long    # or whatever your long partition is called
   ```

2. Submit:
   ```bash
   sbatch scripts/greater_than_5_days/full_pipeline.slurm
   ```
