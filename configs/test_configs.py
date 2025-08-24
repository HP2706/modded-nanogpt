#!/usr/bin/env python3

"""
Test script to verify that the configuration system works correctly.
This script tests loading different model configurations.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from omegaconf import OmegaConf
    import yaml
    print("✓ OmegaConf imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OmegaConf: {e}")
    sys.exit(1)

def test_config_loading():
    """Test loading various configuration files"""
    configs_dir = project_root / "configs"
    
    if not configs_dir.exists():
        print(f"✗ Configs directory not found: {configs_dir}")
        return False
    
    # Test loading the base training config
    training_config_path = configs_dir / "training.yaml"
    if training_config_path.exists():
        try:
            with open(training_config_path, 'r') as f:
                training_config = yaml.safe_load(f)
            print("✓ Training config loaded successfully")
            print(f"  - Train sequence length: {training_config.get('train_seq_len', 'Not specified')}")
            print(f"  - Number of iterations: {training_config.get('num_iterations', 'Not specified')}")
        except Exception as e:
            print(f"✗ Failed to load training config: {e}")
            return False
    else:
        print(f"✗ Training config not found: {training_config_path}")
        return False
    
    # Test loading a few model configs
    model_configs = ["sedd.yaml", "gpt2.yaml", "ngpt.yaml"]
    for config_name in model_configs:
        config_path = configs_dir / config_name
        if config_path.exists():
            try:
                config = OmegaConf.load(str(config_path))
                print(f"✓ {config_name} loaded successfully")
                if 'model_cfg' in config:
                    print(f"  - Model config keys: {list(config.model_cfg.keys())}")
            except Exception as e:
                print(f"✗ Failed to load {config_name}: {e}")
                return False
        else:
            print(f"⚠ {config_name} not found (this may be expected)")
    
    return True

def test_config_inheritance():
    """Test that config inheritance works correctly"""
    configs_dir = project_root / "configs"
    
    # Test a model config that inherits from training
    test_configs = ["sedd.yaml", "gpt2.yaml"]
    
    for config_name in test_configs:
        config_path = configs_dir / config_name
        if config_path.exists():
            try:
                config = OmegaConf.load(str(config_path))
                # Check if defaults are properly set
                if hasattr(config, 'defaults'):
                    print(f"✓ {config_name} has defaults: {config.defaults}")
                else:
                    print(f"⚠ {config_name} doesn't have explicit defaults (may still inherit)")
            except Exception as e:
                print(f"✗ Failed to check inheritance for {config_name}: {e}")
                return False
    
    return True

if __name__ == "__main__":
    print("Testing configuration system...")
    print("=" * 50)
    
    success = True
    success &= test_config_loading()
    print()
    success &= test_config_inheritance()
    
    print("=" * 50)
    if success:
        print("✓ All configuration tests passed!")
        sys.exit(0)
    else:
        print("✗ Some configuration tests failed!")
        sys.exit(1)