# Configuration System

This directory contains the configuration files for all models in the modded-nanogpt project. Each model has its own YAML configuration file that inherits from the base `training.yaml` configuration.

## Structure

- `training.yaml` - Base training configuration that all models inherit from
- `<model_name>.yaml` - Individual model configurations
- `default_model_adapter.py` - Default model adapter that other adapters can inherit from
- `test_configs.py` - Test script to verify configuration loading

## Models with Configurations

- `current_best_gpt.yaml` - Current best GPT model
- `d3pm.yaml` - D3PM (Deep Diffusion Probabilistic Models)
- `gpt2.yaml` - Standard GPT-2 model
- `hnet.yaml` - H-Net (Hierarchical Network)
- `hnet_x_nsa.yaml` - Combination of H-Net and NSA
- `mamba.yaml` - Mamba state space model
- `mdl.yaml` - MDLM (Masked Diffusion Language Model)
- `megabyte.yaml` - MegaByte model
- `mod.yaml` - Mixture of Depths
- `moe.yaml` - Mixture of Experts
- `mtm.yaml` - Multi-Token Model
- `mtp.yaml` - Multi-Token Prediction model
- `ngpt.yaml` - Normalized GPT
- `nsa.yaml` - Native Sparse Attention
- `rd.yaml` - Recurrent Depth
- `sedd.yaml` - SEDD (Score Entropy Discrete Diffusion)
- `vanilla.yaml` - Vanilla Transformer

## Usage

To use a specific model configuration, run the trainer with the `--hydra_model_cfg` flag:

```bash
python mytrainer.py --hydra_model_cfg=configs/sedd.yaml
```

## Configuration Inheritance

All model configurations inherit from `training.yaml` using Hydra's defaults feature. This allows for consistent training parameters across all models while allowing model-specific overrides.

## Adding New Models

To add a new model:

1. Create a new YAML file in this directory with the model name (e.g., `new_model.yaml`)
2. Add `defaults: - training` at the top of the file
3. Define model-specific parameters under the `model_cfg` section
4. Override any training parameters as needed

## Default Model Adapter

The `default_model_adapter.py` file provides a base adapter class that other model adapters can inherit from to reduce code duplication.

## Testing Configurations

To verify that all configurations load correctly, run the test script:

```bash
python configs/test_configs.py
```