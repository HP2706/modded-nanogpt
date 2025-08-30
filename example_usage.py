#!/usr/bin/env python3
"""
Example usage of the new Hydra-based trainer.

Usage examples:

# Use current-best GPT model with default config
python mytrainer.py --config-name=current_best_gpt

# Use NGPT model  
python mytrainer.py --config-name=ngpt

# Use Mamba model
python mytrainer.py --config-name=mamba

# Override specific parameters
python mytrainer.py --config-name=current_best_gpt num_iterations=2000 seq_len=32768

# Override nested model config
python mytrainer.py --config-name=current_best_gpt model_cfg.model_dim=1024

# Use different data files
python mytrainer.py --config-name=current_best_gpt train_files="my_data/train_*.bin"

# Disable wandb and torch compile
python mytrainer.py --config-name=current_best_gpt use_wandb=false torch_compile=false

# Multi-value overrides
python mytrainer.py --config-name=mamba seq_len=2048 num_iterations=1000 use_liger=false
"""

print(__doc__)