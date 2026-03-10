#!/bin/bash -l
#SBATCH --job-name=gled_diff
#SBATCH --output=/home/s/satvikpraveen/Experiments_P5/logs/diff_%j.out
#SBATCH --error=/home/s/satvikpraveen/Experiments_P5/logs/diff_%j.err

#SBATCH -p general
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:TitanV:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Diffusion Model Training - G-LED ExperimentP5
# Trains Imagen-based diffusion model on DICOM/CT data

set -euo pipefail

PROJECT_DIR=/home/s/satvikpraveen/Experiments_P5
ROOT="$PROJECT_DIR"
mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/output"
mkdir -p "$ROOT/checkpoints"

echo "=========================================="
echo "G-LED Diffusion Model Training"
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
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
SEQ_MODEL="${SEQ_MODEL:-}"

echo ""
echo "Starting diffusion model training"
echo "Preset: $PRESET | Batch: $BATCH_SIZE | Epochs: $EPOCHS | Seed: $SEED"
if [ -n "$SEQ_MODEL" ]; then
    echo "Sequential model: $SEQ_MODEL"
fi
echo ""

# Build command
CMD="python $ROOT/train_diffusion.py --preset $PRESET --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --seed $SEED --num_workers 8 --amp"

if [ -n "$SEQ_MODEL" ]; then
    CMD="$CMD --seq_model_path $SEQ_MODEL"
fi

# Run training
eval $CMD

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
