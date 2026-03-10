#!/usr/bin/env python3
"""
Verify ExperimentP5 Installation
================================

Checks that all required components are present and can be imported.
"""

import sys
import os
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ MISSING: {description}")
        return False

def check_dir(path, description):
    """Check if a directory exists and has files."""
    dir_path = Path(path)
    if dir_path.exists() and dir_path.is_dir():
        files = list(dir_path.glob('*.py'))
        if files:
            print(f"✅ {description} ({len(files)} files)")
            return True
        else:
            print(f"⚠️  {description} (directory empty)")
            return False
    else:
        print(f"❌ MISSING: {description}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {description} - {e}")
        return False

print("="*70)
print("ExperimentP5 Installation Verification")
print("="*70)
print()

# Change to ExperimentP5 directory
os.chdir(Path(__file__).parent)

all_ok = True

# Check main scripts
print("Main Scripts:")
all_ok &= check_file('train_sequential.py', 'Sequential training script')
all_ok &= check_file('train_diffusion.py', 'Diffusion training script')
all_ok &= check_file('eval_sequential.py', 'Sequential evaluation script')
all_ok &= check_file('eval_diffusion.py', 'Diffusion evaluation script')
print()

# Check configuration
print("Configuration:")
all_ok &= check_file('configs/base_config.py', 'Configuration system')
all_ok &= check_file('requirements.txt', 'Requirements file')
print()

# Check data loading
print("Data Loading:")
all_ok &= check_file('data/optimized_data.py', 'Optimized data loaders')
print()

# Check trainers
print("Training Infrastructure:")
all_ok &= check_file('train_test_seq/optimized_trainer.py', 'Sequential trainer')
all_ok &= check_file('train_test_spatial/optimized_trainer.py', 'Diffusion trainer')
print()

# Check utilities
print("Utilities:")
all_ok &= check_file('util/optimized_utils.py', 'Utilities (checkpointing, metrics, monitoring)')
print()

# Check model modules
print("Model Components:")
all_ok &= check_dir('mimagen_pytorch', 'IMAGEN diffusion model')
all_ok &= check_dir('transformer', 'Sequential transformer model')
print()

# Check SLURM scripts
print("Cluster Support (SLURM):")
all_ok &= check_file('scripts/train_seq.slurm', 'Sequential SLURM job')
all_ok &= check_file('scripts/train_diff.slurm', 'Diffusion SLURM job')
all_ok &= check_file('scripts/full_pipeline.slurm', 'Pipeline SLURM job')
all_ok &= check_file('scripts/debug.slurm', 'Debug SLURM job')
all_ok &= check_file('scripts/submit_job.sh', 'Job submission helper')
print()

# Check documentation
print("Documentation:")
all_ok &= check_file('README.md', 'Main documentation')
all_ok &= check_file('QUICKSTART.md', 'Quick start guide')
all_ok &= check_file('SUMMARY.md', 'Summary document')
all_ok &= check_file('IMPLEMENTATION_COMPLETE.md', 'Completion status')
print()

# Check examples
print("Examples:")
all_ok &= check_file('examples/custom_config.py', 'Custom configuration example')
all_ok &= check_file('examples/monitor_training.py', 'Training monitoring example')
print()

# Try importing key modules
print("Module Imports:")
sys.path.insert(0, str(Path.cwd()))
all_ok &= check_import('configs.base_config', 'Configuration module')
all_ok &= check_import('data.optimized_data', 'Data loading module')
all_ok &= check_import('util.optimized_utils', 'Utilities module')
all_ok &= check_import('transformer.sequentialModel', 'Sequential model')
all_ok &= check_import('mimagen_pytorch', 'IMAGEN module')
print()

# Try loading a preset config
try:
    from configs.base_config import get_preset_config
    config = get_preset_config('optimized_default')
    print("✅ Configuration presets work")
except Exception as e:
    print(f"❌ Configuration presets error: {e}")
    all_ok = False
print()

print("="*70)
if all_ok:
    print("✅ ALL CHECKS PASSED - ExperimentP5 is ready to use!")
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Quick test: python train_sequential.py --debug")
    print("  3. See QUICKSTART.md for usage examples")
else:
    print("⚠️  SOME CHECKS FAILED - Please review errors above")
    print()
    print("Try:")
    print("  - Re-running this script")
    print("  - Checking file permissions")
    print("  - Installing dependencies: pip install -r requirements.txt")

print("="*70)

sys.exit(0 if all_ok else 1)
