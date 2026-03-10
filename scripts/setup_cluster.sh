#!/bin/bash
# Comprehensive Cluster Setup Script
# Run this after transferring ExperimentP5 to cluster

set -e  # Exit on error

echo "=================================================="
echo "ExperimentP5 Cluster Setup"
echo "=================================================="
echo ""

# 1. Check Python version
echo "1. Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11) ]]; then
    echo "   ⚠️  Warning: Python 3.9-3.11 recommended"
fi
echo ""

# 2. Create virtual environment
echo "2. Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi
echo ""

# 3. Activate and upgrade pip
echo "3. Activating environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
echo "   ✅ Pip upgraded"
echo ""

# 4. Install dependencies
echo "4. Installing dependencies from requirements.txt..."
echo "   This may take 5-10 minutes..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"
echo ""

# 5. Check PyTorch and CUDA
echo "5. Verifying PyTorch and CUDA..."
python -c "import torch; print(f'   PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'   GPU count: {torch.cuda.device_count()}')" 
echo ""

# 6. Create necessary directories
echo "6. Creating output directories..."
mkdir -p output logs checkpoints
echo "   ✅ Directories created"
echo ""

# 7. Make scripts executable
echo "7. Making scripts executable..."
chmod +x scripts/*.slurm scripts/*.sh train_sequential.py train_diffusion.py
chmod +x eval_sequential.py eval_diffusion.py verify_installation.py
echo "   ✅ Scripts executable"
echo ""

# 8. Run verification
echo "8. Running installation verification..."
echo ""
python verify_installation.py
VERIFY_STATUS=$?
echo ""

# 9. Summary
echo "=================================================="
if [ $VERIFY_STATUS -eq 0 ]; then
    echo "✅ SETUP COMPLETE - ExperimentP5 ready to use!"
    echo ""
    echo "Next steps:"
    echo "  1. Update data paths in configs/base_config.py"
    echo "  2. Test: python train_sequential.py --debug"
    echo "  3. Submit job: ./scripts/submit_job.sh seq"
else
    echo "⚠️  SETUP INCOMPLETE - Please review errors above"
fi
echo "=================================================="
