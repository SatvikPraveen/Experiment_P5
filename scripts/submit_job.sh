#!/bin/bash
# Quick job submission helper script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

print_usage() {
    echo "Usage: ./submit_job.sh [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  seq              Train sequential model only"
    echo "  diff             Train diffusion model only"
    echo "  pipeline         Train full pipeline (seq then diff)"
    echo "  debug            Quick debug test"
    echo ""
    echo "Examples:"
    echo "  ./submit_job.sh seq PRESET=720_816_medium_unet"
    echo "  ./submit_job.sh diff BATCH_SIZE=16"
    echo "  ./submit_job.sh pipeline PRESET=optimized_default"
    echo "  ./submit_job.sh debug"
    echo ""
    echo "Environment variables:"
    echo "  PRESET           Configuration preset (default: optimized_default)"
    echo "  BATCH_SIZE       Batch size"
    echo "  EPOCHS           Number of epochs"
    echo "  LR               Learning rate"
    echo "  SEQ_MODEL        Path to sequential model (for diff mode)"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

MODE=$1
shift  # Remove first argument

# Export remaining arguments as environment variables
for arg in "$@"; do
    export "$arg"
done

# Submit appropriate job
case $MODE in
    seq)
        echo -e "${GREEN}Submitting sequential training job...${NC}"
        sbatch "$SCRIPT_DIR/train_seq.slurm"
        ;;
    diff)
        echo -e "${GREEN}Submitting diffusion training job...${NC}"
        sbatch "$SCRIPT_DIR/train_diff.slurm"
        ;;
    pipeline)
        echo -e "${GREEN}Submitting full pipeline job (requires >5 day partition)...${NC}"
        sbatch "$SCRIPT_DIR/greater_than_5_days/full_pipeline.slurm"
        ;;
    debug)
        echo -e "${YELLOW}Submitting debug job...${NC}"
        sbatch "$SCRIPT_DIR/debug.slurm"
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        print_usage
        exit 1
        ;;
esac

# Show queue
echo ""
echo "Current queue:"
squeue -u $USER
