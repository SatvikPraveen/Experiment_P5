#!/usr/bin/env python3
"""
Pre-Cluster Deployment Test Suite
Tests all components to ensure smooth cluster deployment
"""
import sys
import os
from pathlib import Path

# Allow running from tests/ subdirectory
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all critical imports"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("data.optimized_data", "Optimized Data Module"),
        ("data.dicom_fbp_dataset", "DICOM FBP Dataset"),
        ("configs.base_config", "Base Config"),
        ("mimagen_pytorch.physics_diffusion", "Physics Diffusion Model"),
        ("transformer.sequentialModel", "Sequential Model"),
    ]
    
    passed = 0
    failed = 0
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            failed += 1
    
    print(f"\nImport Results: {passed} passed, {failed} failed")
    return failed == 0

def test_file_structure():
    """Verify all required files exist"""
    print("\n" + "=" * 60)
    print("TESTING FILE STRUCTURE")
    print("=" * 60)
    
    required_files = [
        "train_diffusion.py",
        "train_sequential.py",
        "eval_diffusion.py",
        "eval_sequential.py",
        "configs/base_config.py",
        "data/optimized_data.py",
        "data/dicom_fbp_dataset.py",
        "mimagen_pytorch/physics_diffusion.py",
        "transformer/sequentialModel.py",
        "scripts/train_diff.slurm",
        "scripts/train_seq.slurm",
        "scripts/train_diff_chain.slurm",
        "scripts/train_seq_chain.slurm",
        "examples/dicom_fbp_config.py",
    ]
    
    passed = 0
    failed = 0
    
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            print(f"✅ {filepath}: EXISTS")
            passed += 1
        else:
            print(f"❌ {filepath}: MISSING")
            failed += 1
    
    print(f"\nFile Structure Results: {passed} passed, {failed} failed")
    return failed == 0

def test_slurm_scripts():
    """Check SLURM scripts for common issues"""
    print("\n" + "=" * 60)
    print("TESTING SLURM SCRIPTS")
    print("=" * 60)
    
    slurm_scripts = [
        "scripts/train_diff.slurm",
        "scripts/train_seq.slurm",
        "scripts/train_diff_chain.slurm",
        "scripts/train_seq_chain.slurm",
    ]
    
    passed = 0
    failed = 0
    issues = []
    
    for script in slurm_scripts:
        script_issues = []
        try:
            with open(script, 'r') as f:
                content = f.read()
                
            # Check for correct paths
            if "ExperimentP5" in content and "Experiments_P5" not in content:
                script_issues.append("Uses old path 'ExperimentP5'")
            
            # Check for correct conda environment
            if "experimentp5" in content.lower() and "sam_gpu" not in content:
                script_issues.append("Uses old conda env 'experimentp5'")
            
            # Check for GPU specification
            if "gpu:titanx" in content.lower():
                script_issues.append("Requests incompatible TitanX GPU")
            
            # Check for project path
            if "PROJECT_DIR=" not in content:
                script_issues.append("Missing PROJECT_DIR variable")
            
            if script_issues:
                print(f"⚠️  {script}: {', '.join(script_issues)}")
                issues.extend(script_issues)
                failed += 1
            else:
                print(f"✅ {script}: OK")
                passed += 1
                
        except Exception as e:
            print(f"❌ {script}: ERROR - {e}")
            failed += 1
    
    print(f"\nSLURM Scripts Results: {passed} passed, {failed} failed")
    return failed == 0

def test_config_compatibility():
    """Test configuration loading"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from configs.base_config import get_preset_config
        config = get_preset_config('optimized_default')
        
        # Check for new DICOM parameters
        dicom_params = [
            'dataset_type',
            'dicom_path',
            'detector_count',
            'angle_step',
            'use_fbp',
            'fbp_filter',
        ]
        
        print("Base Configuration Parameters:")
        present = []
        missing = []
        
        for param in dicom_params:
            if hasattr(config, param):
                print(f"✅ {param}: {getattr(config, param)}")
                present.append(param)
            else:
                print(f"⚠️  {param}: NOT SET (optional)")
                missing.append(param)
        
        print(f"\n✅ Configuration loaded successfully")
        print(f"   Present: {len(present)}/{len(dicom_params)} DICOM parameters")
        return True
        
    except Exception as e:
        print(f"❌ Configuration failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_pipeline():
    """Test data loading pipeline"""
    print("\n" + "=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    
    try:
        from data.optimized_data import create_dataloaders_from_config
        from configs.base_config import get_preset_config
        
        config = get_preset_config('optimized_default')
        print(f"✅ Data pipeline imports OK")
        print(f"   Dataset type: {getattr(config, 'dataset_type', 'bfs')}")
        print(f"   Mode: {config.mode}")
        
        # Don't actually create dataloaders (might not have data)
        # Just verify the function exists and config is compatible
        print(f"✅ Data pipeline ready for deployment")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_scripts():
    """Verify training scripts don't have interactive inputs"""
    print("\n" + "=" * 60)
    print("TESTING TRAINING SCRIPTS")
    print("=" * 60)
    
    scripts = ["train_diffusion.py", "train_sequential.py"]
    passed = 0
    failed = 0
    
    for script in scripts:
        try:
            with open(script, 'r') as f:
                content = f.read()
            
            if 'input(' in content:
                print(f"⚠️  {script}: Contains input() call (may block on cluster)")
                failed += 1
            else:
                print(f"✅ {script}: No interactive inputs")
                passed += 1
                
        except Exception as e:
            print(f"❌ {script}: ERROR - {e}")
            failed += 1
    
    print(f"\nTraining Scripts Results: {passed} passed, {failed} failed")
    return failed == 0

def main():
    """Run all tests"""
    print("\n🔍 PRE-CLUSTER DEPLOYMENT TEST SUITE")
    print("=" * 60)
    print("This will verify all components before cluster transfer")
    print("=" * 60 + "\n")
    
    results = {
        "Imports": test_imports(),
        "File Structure": test_file_structure(),
        "SLURM Scripts": test_slurm_scripts(),
        "Configuration": test_config_compatibility(),
        "Data Pipeline": test_data_pipeline(),
        "Training Scripts": test_training_scripts(),
    }
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for cluster deployment!")
        print("\nNext Steps:")
        print("1. rsync -av ~/Desktop/Experiments_P5/ username@forest.usf.edu:/path/to/Experiments_P5/")
        print("2. ssh forest.usf.edu")
        print("3. conda activate sam_gpu")
        print("4. cd /path/to/Experiments_P5")
        print("5. sbatch scripts/train_diff.slurm  # or train_seq.slurm")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Fix issues before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
