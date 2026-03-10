#!/usr/bin/env python3
"""
Quick Pre-Deployment Test
==========================

Validates all fixes and configuration before cluster deployment.
Run this locally before transferring to cluster.
"""

import sys
import os
from pathlib import Path

# Allow running from tests/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from configs.base_config import get_preset_config, ExperimentConfig
        from data.optimized_data import BFSDataset
        from transformer.sequentialModel import SequentialModel
        from train_test_seq.optimized_trainer import OptimizedSequentialTrainer
        from util.optimized_utils import set_random_seed
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_all_presets():
    """Test that all preset configurations can be loaded."""
    print("\nTesting preset configurations...")
    from configs.base_config import get_preset_config
    
    presets = [
        'baseline', '720_816_standard', '720_816_high_unet',
        '720_816_medium_unet', '720_1024_large_unet', 
        'updated_1024', 'feather_720_432', 'optimized_default'
    ]
    
    all_ok = True
    for preset in presets:
        try:
            config = get_preset_config(preset)
            
            # Validate dimension consistency
            expected_embd = config.model.coarse_dim[0] * config.model.coarse_dim[1] * 2
            if expected_embd != config.model.n_embd:
                print(f"  ❌ {preset}: dimension mismatch (coarse_dim={config.model.coarse_dim}, n_embd={config.model.n_embd})")
                all_ok = False
            else:
                print(f"  ✅ {preset}: {config.data.image_height}×{config.data.image_width}, dims valid")
        except Exception as e:
            print(f"  ❌ {preset}: {e}")
            all_ok = False
    
    return all_ok

def test_trainer_dimensions():
    """Test that trainer uses config dimensions (not hardcoded)."""
    print("\nTesting trainer dimension handling...")
    
    # Read the trainer file
    trainer_file = Path(__file__).parent / 'train_test_seq' / 'optimized_trainer.py'
    
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Check that hardcoded dimensions are gone
    if 'reshape([b_size * num_time, num_velocity, 512, 512])' in content:
        print("  ❌ Found hardcoded 512, 512 dimensions!")
        print("     This will cause crashes on non-baseline configs")
        return False
    
    # Check that config dimensions are used
    if 'self.config.data.image_height' in content and 'self.config.data.image_width' in content:
        print("  ✅ Trainer uses config dimensions correctly")
        return True
    else:
        print("  ⚠️  Could not verify dimension usage")
        return False

def test_validation_functions():
    """Test that validation functions exist."""
    print("\nTesting validation functions...")
    
    # Check train_sequential.py
    seq_file = Path(__file__).parent / 'train_sequential.py'
    with open(seq_file, 'r') as f:
        seq_content = f.read()
    
    if '_validate_config' in seq_content:
        print("  ✅ Sequential validation function exists")
        seq_ok = True
    else:
        print("  ❌ Sequential validation function missing")
        seq_ok = False
    
    # Check train_diffusion.py
    diff_file = Path(__file__).parent / 'train_diffusion.py'
    with open(diff_file, 'r') as f:
        diff_content = f.read()
    
    if '_validate_config' in diff_content:
        print("  ✅ Diffusion validation function exists")
        diff_ok = True
    else:
        print("  ❌ Diffusion validation function missing")
        diff_ok = False
    
    return seq_ok and diff_ok

def test_slurm_scripts():
    """Test that SLURM scripts have improved module loading."""
    print("\nTesting SLURM script improvements...")
    
    scripts = ['train_seq.slurm', 'train_diff.slurm', 'full_pipeline.slurm', 'debug.slurm']
    all_ok = True
    
    for script in scripts:
        script_file = Path(__file__).parent / 'scripts' / script
        if not script_file.exists():
            print(f"  ❌ {script}: not found")
            all_ok = False
            continue
        
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Check for improvements
        if 'UNCOMMENT and adjust' in content and 'if [ -d "${EXPERIMENT_DIR}/venv"' in content:
            print(f"  ✅ {script}: has improved module loading")
        else:
            print(f"  ⚠️  {script}: may need module loading improvements")
            all_ok = False
    
    return all_ok

def test_documentation():
    """Test that documentation files exist."""
    print("\nTesting documentation...")
    
    docs = {
        'CLUSTER_CHECKLIST.md': 'Deployment checklist',
        'FIXES_APPLIED.md': 'Fixes documentation',
        'README.md': 'Main readme',
        'QUICKSTART.md': 'Quick start guide'
    }
    
    all_ok = True
    for doc, desc in docs.items():
        doc_file = Path(__file__).parent / doc
        if doc_file.exists():
            print(f"  ✅ {doc}: {desc}")
        else:
            print(f"  ❌ {doc}: missing")
            all_ok = False
    
    return all_ok

def main():
    print("="*70)
    print("ExperimentP5 Pre-Deployment Validation")
    print("="*70)
    print()
    
    tests = [
        ("Module imports", test_imports),
        ("Preset configurations", test_all_presets),
        ("Trainer dimensions", test_trainer_dimensions),
        ("Config validation", test_validation_functions),
        ("SLURM scripts", test_slurm_scripts),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for cluster deployment!")
        print("\nNext steps:")
        print("  1. Review CLUSTER_CHECKLIST.md")
        print("  2. Configure data paths in configs/base_config.py")
        print("  3. Transfer to cluster and run setup_cluster.sh")
        print("  4. Test with: ./scripts/submit_job.sh debug")
        return 0
    else:
        print("\n⚠️  Some tests failed - please review and fix before deployment")
        return 1

if __name__ == '__main__':
    sys.exit(main())
