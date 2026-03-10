#!/bin/bash
# Submit all 7 preset configurations for full convergence study
# Each will automatically chain jobs as needed

set -euo pipefail

ROOT=/home/s/satvikpraveen/Experiments_P5
SCRIPT_DIR="$ROOT/scripts"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "G-LED: Full Reproducibility Study"
echo "=========================================="
echo "Submitting all 7 preset configurations"
echo "Each will train to full convergence (500 epochs)"
echo "Jobs will auto-resubmit every 23 hours"
echo "=========================================="
echo ""

# All 7 presets from the paper
PRESETS=(
    "baseline"
    "720_816_standard"
    "720_816_high_unet"
    "720_816_medium_unet"
    "720_1024_large_unet"
    "updated_1024"
    "feather_720_432"
    "optimized_default"
)

# Random seeds for reproducibility
SEED=42

# Training mode: seq, diff, or both
MODE="${1:-both}"  # Default: train both models

echo -e "${YELLOW}Training mode: $MODE${NC}"
echo ""

for PRESET in "${PRESETS[@]}"; do
    echo "----------------------------------------"
    echo -e "${GREEN}Submitting: $PRESET${NC}"
    
    if [ "$MODE" == "seq" ] || [ "$MODE" == "both" ]; then
        # Submit sequential training
        SEQ_JOB=$(sbatch --parsable \
            --export=PRESET="$PRESET",TARGET_EPOCHS=500,SEED="$SEED" \
            "$SCRIPT_DIR/train_seq_chain.slurm")
        echo "  Sequential job: $SEQ_JOB"
    fi
    
    if [ "$MODE" == "diff" ] || [ "$MODE" == "both" ]; then
        # For diffusion, optionally wait for sequential to finish
        if [ "$MODE" == "both" ]; then
            # Submit diffusion with dependency on sequential
            DIFF_JOB=$(sbatch --parsable \
                --dependency=afterok:$SEQ_JOB \
                --export=PRESET="$PRESET",TARGET_EPOCHS=500,SEED="$SEED" \
                "$SCRIPT_DIR/train_diff_chain.slurm")
            echo "  Diffusion job: $DIFF_JOB (waits for $SEQ_JOB)"
        else
            # Submit diffusion independently
            DIFF_JOB=$(sbatch --parsable \
                --export=PRESET="$PRESET",TARGET_EPOCHS=500,SEED="$SEED" \
                "$SCRIPT_DIR/train_diff_chain.slurm")
            echo "  Diffusion job: $DIFF_JOB"
        fi
    fi
    
    echo ""
    sleep 1  # Small delay between submissions
done

echo "=========================================="
echo "✅ All configurations submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f $ROOT/logs/*.out"
echo ""
echo "Check wandb dashboard:"
echo "  https://wandb.ai/"
echo "=========================================="
