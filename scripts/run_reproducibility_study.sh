#!/bin/bash
# Run all experiments with different seeds for reproducibility study

cat << 'EOF'
================================================
ExperimentP5: Reproducibility Experiment Runner
================================================

This script will run all 7 configurations with multiple seeds
to verify reproducibility and gather statistics.

Configuration:
- 7 presets (all branches)
- 3 random seeds (42, 123, 456)
- Total: 21 sequential experiments
- Optionally: 21 diffusion experiments

Estimated time: 
  Sequential: ~70-140 hours (depends on config)
  Diffusion: ~100-200 hours
  
On 4+ GPUs in parallel: Can complete in 1-2 days
EOF

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Configuration
PRESETS=(
    "baseline"
    "720_816_standard"
    "720_816_high_unet"
    "720_816_medium_unet"
    "updated_1024"
    "feather_720_432"
    "optimized_default"
)

SEEDS=(42 123 456)
RUN_DIFFUSION=${RUN_DIFFUSION:-false}

# Create results directory
RESULTS_DIR="reproducibility_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "Results will be saved in: $RESULTS_DIR"
echo ""

# Function to check if SLURM is available
if command -v sbatch &> /dev/null; then
    USE_SLURM=true
    echo "✅ SLURM detected - using cluster submission"
else
    USE_SLURM=false
    echo "ℹ️  No SLURM - using local execution"
fi

echo ""
echo "================================================"
echo "PHASE 1: Sequential Model Training"
echo "================================================"
echo ""

JOB_COUNT=0

for preset in "${PRESETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        EXP_NAME="${preset}_seed${seed}"
        echo "Submitting: $EXP_NAME"
        
        if [ "$USE_SLURM" = true ]; then
            # SLURM submission
            JOB_ID=$(sbatch --parsable \
                --job-name="seq_${EXP_NAME}" \
                --output="${RESULTS_DIR}/seq_${EXP_NAME}_%j.out" \
                --error="${RESULTS_DIR}/seq_${EXP_NAME}_%j.err" \
                scripts/train_seq.slurm \
                PRESET="$preset" \
                SEED="$seed" \
                NAME="$EXP_NAME")
            
            echo "  Job ID: $JOB_ID"
            echo "$preset,$seed,$JOB_ID,sequential,submitted" >> "$RESULTS_DIR/job_tracker.csv"
        else
            # Local execution (background)
            python train_sequential.py \
                --preset "$preset" \
                --seed "$seed" \
                --name "$EXP_NAME" \
                > "${RESULTS_DIR}/seq_${EXP_NAME}.out" 2>&1 &
            
            LOCAL_PID=$!
            echo "  PID: $LOCAL_PID"
            echo "$preset,$seed,$LOCAL_PID,sequential,running" >> "$RESULTS_DIR/job_tracker.csv"
        fi
        
        ((JOB_COUNT++))
        sleep 2  # Avoid overwhelming scheduler
    done
done

echo ""
echo "✅ Submitted $JOB_COUNT sequential training jobs"
echo ""

# Diffusion training (optional)
if [ "$RUN_DIFFUSION" = true ]; then
    echo ""
    echo "================================================"
    echo "PHASE 2: Diffusion Model Training"
    echo "================================================"
    echo ""
    echo "⚠️  Note: This will wait for sequential models to complete"
    echo ""
    
    DIFF_COUNT=0
    
    for preset in "${PRESETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            EXP_NAME="${preset}_seed${seed}"
            SEQ_CHECKPOINT="output/${EXP_NAME}_*/checkpoints/best.pt"
            
            echo "Scheduling diffusion: $EXP_NAME"
            
            if [ "$USE_SLURM" = true ]; then
                # Submit with dependency on sequential job
                SEQ_JOB_ID=$(grep "$preset,$seed" "$RESULTS_DIR/job_tracker.csv" | grep "sequential" | cut -d',' -f3)
                
                JOB_ID=$(sbatch --parsable \
                    --dependency=afterok:$SEQ_JOB_ID \
                    --job-name="diff_${EXP_NAME}" \
                    --output="${RESULTS_DIR}/diff_${EXP_NAME}_%j.out" \
                    --error="${RESULTS_DIR}/diff_${EXP_NAME}_%j.err" \
                    scripts/train_diff.slurm \
                    PRESET="$preset" \
                    SEED="$seed" \
                    SEQ_MODEL="$SEQ_CHECKPOINT" \
                    NAME="diff_${EXP_NAME}")
                
                echo "  Job ID: $JOB_ID (depends on $SEQ_JOB_ID)"
                echo "$preset,$seed,$JOB_ID,diffusion,submitted,depends:$SEQ_JOB_ID" >> "$RESULTS_DIR/job_tracker.csv"
            fi
            
            ((DIFF_COUNT++))
            sleep 2
        done
    done
    
    echo ""
    echo "✅ Submitted $DIFF_COUNT diffusion training jobs"
fi

echo ""
echo "================================================"
echo "SUMMARY"
echo "================================================"
echo "Sequential jobs: $JOB_COUNT"
if [ "$RUN_DIFFUSION" = true ]; then
    echo "Diffusion jobs: $DIFF_COUNT"
fi
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Monitor progress:"
if [ "$USE_SLURM" = true ]; then
    echo "  squeue -u \$USER"
    echo "  tail -f ${RESULTS_DIR}/*.out"
else
    echo "  jobs"
    echo "  tail -f ${RESULTS_DIR}/*.out"
fi
echo ""
echo "Job tracker: ${RESULTS_DIR}/job_tracker.csv"
echo "================================================"
