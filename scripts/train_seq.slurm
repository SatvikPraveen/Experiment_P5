#!/bin/bash -l
#SBATCH --job-name=gled_seq
#SBATCH --output=/home/s/satvikpraveen/Experiments_P5/logs/seq_%j.out
#SBATCH --error=/home/s/satvikpraveen/Experiments_P5/logs/seq_%j.err

#SBATCH -p general
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:TitanV:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Sequential Model Training - G-LED ExperimentP5
# Trains GPT-2 style transformer on BFS dataset

set -euo pipefail

PROJECT_DIR=/home/s/satvikpraveen/Experiments_P5
ROOT="$PROJECT_DIR"
mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/output"
mkdir -p "$ROOT/checkpoints"

echo "=========================================="
echo "G-LED Sequential Model Training"
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

# Training parameters
PRESET="${PRESET:-optimized_default}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

echo ""
echo "Starting sequential model training"
echo "Preset: $PRESET | Batch: $BATCH_SIZE | Epochs: $EPOCHS | Seed: $SEED"
echo ""

# Run training
python "$ROOT/train_sequential.py" \
    --preset "$PRESET" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_workers 8 \
    --amp

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
