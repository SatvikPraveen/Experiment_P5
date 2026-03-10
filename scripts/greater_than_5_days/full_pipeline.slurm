#!/bin/bash -l
#SBATCH --job-name=gled_pipeline
#SBATCH --output=/home/s/satvikpraveen/Experiments_P5/logs/pipeline_%j.out
#SBATCH --error=/home/s/satvikpraveen/Experiments_P5/logs/pipeline_%j.err

#SBATCH -p Quick
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Full Pipeline Training - G-LED ExperimentP5
# Trains both sequential and diffusion models end-to-end

set -euo pipefail

ROOT=/home/s/satvikpraveen/Experiments_P5
mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/output"
mkdir -p "$ROOT/checkpoints"

echo "=========================================="
echo "G-LED Full Training Pipeline"
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
SEQ_BATCH_SIZE="${SEQ_BATCH_SIZE:-16}"
DIFF_BATCH_SIZE="${DIFF_BATCH_SIZE:-8}"
SEQ_EPOCHS="${SEQ_EPOCHS:-100}"
DIFF_EPOCHS="${DIFF_EPOCHS:-100}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

echo ""
echo "Configuration:"
echo "  Preset: $PRESET"
echo "  Sequential: batch=$SEQ_BATCH_SIZE, epochs=$SEQ_EPOCHS"
echo "  Diffusion: batch=$DIFF_BATCH_SIZE, epochs=$DIFF_EPOCHS"
echo "  Learning rate: $LR | Seed: $SEED"
echo "=========================================="

# ============================================
# STAGE 1: Train Sequential Model
# ============================================
echo ""
echo "=========================================="
echo "STAGE 1: Training Sequential Model"
echo "=========================================="
echo ""

python "$ROOT/train_sequential.py" \
    --preset "$PRESET" \
    --batch_size "$SEQ_BATCH_SIZE" \
    --epochs "$SEQ_EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_workers 8 \
    --amp \
    --multi_gpu

SEQ_EXIT_CODE=$?

if [ $SEQ_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Sequential training failed with exit code $SEQ_EXIT_CODE"
    exit $SEQ_EXIT_CODE
fi

# Find the best sequential model
SEQ_MODEL_PATH=$(find "$ROOT/output" -name "best.pt" -path "*seq*" | head -n 1)

if [ -z "$SEQ_MODEL_PATH" ]; then
    echo "WARNING: Could not find best sequential model, looking for latest..."
    SEQ_MODEL_PATH=$(find "$ROOT/output" -name "*.pt" -path "*seq*" | sort -r | head -n 1)
fi

if [ -z "$SEQ_MODEL_PATH" ]; then
    echo "ERROR: No sequential model checkpoint found!"
    exit 1
fi

echo ""
echo "✅ Sequential model trained successfully!"
echo "Model: $SEQ_MODEL_PATH"
echo ""

# ============================================
# STAGE 2: Train Diffusion Model
# ============================================
echo ""
echo "==========================================="
echo "STAGE 2: Training Diffusion Model"
echo "==========================================="
echo ""

python "$ROOT/train_diffusion.py" \
    --preset "$PRESET" \
    --batch_size "$DIFF_BATCH_SIZE" \
    --epochs "$DIFF_EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_workers 8 \
    --amp \
    --multi_gpu \
    --seq_model_path "$SEQ_MODEL_PATH"

DIFF_EXIT_CODE=$?

if [ $DIFF_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Diffusion training failed with exit code $DIFF_EXIT_CODE"
    exit $DIFF_EXIT_CODE
fi

# Find the best diffusion model
DIFF_MODEL_PATH=$(find "$ROOT/output" -name "*.pt" -path "*diff*" | sort -r | head -n 1)

echo ""
echo "✅ Diffusion model trained successfully!"
echo "Model: $DIFF_MODEL_PATH"
echo ""

# ============================================
# SUCCESS
# ============================================
echo ""
echo "==========================================="
echo "FULL PIPELINE COMPLETE ✅"
echo "==========================================="
echo "Sequential: $SEQ_MODEL_PATH"
echo "Diffusion: $DIFF_MODEL_PATH"
echo "End time: $(date)"
echo "==========================================="

exit 0
