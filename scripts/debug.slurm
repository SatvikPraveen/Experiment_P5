#!/bin/bash -l
#SBATCH --job-name=gled_debug
#SBATCH --output=/home/s/satvikpraveen/Experiments_P5/logs/debug_%j.out
#SBATCH --error=/home/s/satvikpraveen/Experiments_P5/logs/debug_%j.err

#SBATCH -p general
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:TitanV:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Debug Mode - G-LED ExperimentP5
# Quick testing and validation before full deployment

set -euo pipefail

ROOT=/home/s/satvikpraveen/Experiments_P5
mkdir -p "$ROOT/logs"

echo "=========================================="
echo "G-LED Debug Mode"
echo "=========================================="
echo "JobID=$SLURM_JOB_ID Node=$SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate sam_gpu

# Verify GPU
nvidia-smi || true

# Change to project root directory
cd "$ROOT"

echo ""
echo "Testing Sequential Model..."
echo ""

python "$ROOT/train_sequential.py" \
    --preset optimized_default \
    --debug \
    --num_workers 4

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Sequential debug failed!"
    exit $EXIT_CODE
fi

echo ""
echo "✅ Sequential test passed"
echo ""
echo "Testing Diffusion Model..."
echo ""

python "$ROOT/train_diffusion.py" \
    --preset optimized_default \
    --debug \
    --num_workers 4

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Debug test completed: exit code $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
