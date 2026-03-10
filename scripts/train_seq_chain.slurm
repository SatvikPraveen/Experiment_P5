#!/bin/bash -l
#SBATCH --job-name=gled_seq_chain
#SBATCH --output=/home/s/satvikpraveen/Experiments_P5/logs/seq_chain_%j.out
#SBATCH --error=/home/s/satvikpraveen/Experiments_P5/logs/seq_chain_%j.err

#SBATCH -p general
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:TitanV:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Sequential Model Training with Auto-Resubmission - G-LED ExperimentP5
# Automatically chains jobs for full convergence beyond 23 hours

set -euo pipefail

PROJECT_DIR=/home/s/satvikpraveen/Experiments_P5
ROOT="$PROJECT_DIR"
mkdir -p "$ROOT/logs"
mkdir -p "$ROOT/output"
mkdir -p "$ROOT/checkpoints"

echo "=========================================="
echo "G-LED Sequential Chained Training"
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
TARGET_EPOCHS="${TARGET_EPOCHS:-500}"  # Full convergence
LR="${LR:-1e-4}"
SEED="${SEED:-42}"

# Checkpoint management
CHECKPOINT_DIR="$ROOT/output/${PRESET}_seq_$(date +%Y%m%d)"
RESUME_CKPT="$CHECKPOINT_DIR/interrupted.pt"

echo ""
echo "Configuration:"
echo "  Preset: $PRESET"
echo "  Target epochs: $TARGET_EPOCHS"
echo "  Checkpoint dir: $CHECKPOINT_DIR"

# Check if resuming from previous job
if [ -f "$RESUME_CKPT" ]; then
    echo "  📥 Resuming from checkpoint: $RESUME_CKPT"
    RESUME_ARG="--resume $RESUME_CKPT"
else
    echo "  🆕 Starting fresh training"
    RESUME_ARG=""
fi
echo "=========================================="
echo ""

# Run training (will save checkpoint on interruption)
python "$ROOT/train_sequential.py" \
    --preset "$PRESET" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$TARGET_EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --num_workers 8 \
    --amp \
    $RESUME_ARG

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Check if training completed or was interrupted
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    exit 0
elif [ -f "$RESUME_CKPT" ]; then
    # Training was interrupted but checkpoint exists - resubmit
    echo "⏰ Job time limit reached. Checkpoint saved."
    echo "🔄 Resubmitting continuation job..."
    
    # Resubmit this same script with same parameters
    NEW_JOB_ID=$(sbatch --parsable \
        --export=PRESET="$PRESET",BATCH_SIZE="$BATCH_SIZE",TARGET_EPOCHS="$TARGET_EPOCHS",LR="$LR",SEED="$SEED" \
        "$ROOT/scripts/train_seq_chain.slurm")
    
    echo "📝 Submitted continuation job: $NEW_JOB_ID"
    exit 0
else
    echo "❌ Training failed without checkpoint"
    exit $EXIT_CODE
fi
