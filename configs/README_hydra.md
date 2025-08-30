# Hydra Configuration System

The trainer now uses [Hydra](https://hydra.cc/) for configuration management instead of command-line arguments. This provides better organization, reusability, and flexibility.

## Structure

- **`data.yaml`**: Data loading and processing configuration
- **`training.yaml`**: Training hyperparameters and settings  
- **Model configs** (e.g., `current_best_gpt.yaml`, `mamba.yaml`): Model-specific settings that inherit from both data and training configs

## Usage

### Basic Usage

```bash
# Train with current-best GPT model
python mytrainer.py --config-name=current_best_gpt

# Train with Mamba model
python mytrainer.py --config-name=mamba

# Train with NGPT model
python mytrainer.py --config-name=ngpt
```

### Override Parameters

```bash
# Override training iterations and sequence length
python mytrainer.py --config-name=current_best_gpt num_iterations=2000 seq_len=32768

# Override nested model configuration
python mytrainer.py --config-name=current_best_gpt model_cfg.model_dim=1024

# Override data files
python mytrainer.py --config-name=current_best_gpt train_files="my_data/train_*.bin"

# Multiple overrides
python mytrainer.py --config-name=mamba seq_len=2048 num_iterations=1000 use_liger=false torch_compile=false
```

### Available Model Types

The following model types are registered in `trainer_registry.py`:

- `current-best`: Current best GPT architecture
- `ngpt`: NGPT model
- `mamba`: Mamba state space model
- `megabyte`: MegaByte hierarchical model
- `mixture-of-depths`: Mixture of Depths transformer
- `mixture-of-experts`: Mixture of Experts model
- `mdlm`: Masked Diffusion Language Model
- `d3pm`: Discrete Diffusion Probabilistic Model
- `recurrent-depth`: Recurrent Depth transformer
- `vanilla`: Vanilla transformer
- `deepseek-mtp`: DeepSeek MTP model
- `base-mtp`: Base MTP model
- `nsa`: NSA model
- `gpt2`: GPT-2 model
- `sedd`: SEDD model

## Creating New Model Configs

1. Create a new YAML file in `configs/` (e.g., `my_model.yaml`)
2. Use this template:

```yaml
# My Model configuration
defaults:
  - data
  - training
  - _self_

# Model type (must be registered in trainer_registry.py)
type: "my-model"

# Model-specific configuration
model_cfg:
  vocab_size: 50257
  num_layers: 12
  model_dim: 768
  # ... other model-specific parameters

# Override data/training parameters if needed
seq_len: 4096
num_iterations: 1500
```

3. Register the model adapter in `trainer_registry.py`:

```python
ADAPTER_PATHS["my-model"] = "models.MyModel:MyModelAdapter"
```

## Configuration Hierarchy

1. **Base configs** (`data.yaml`, `training.yaml`) provide defaults
2. **Model configs** inherit from base configs and can override any parameter
3. **Command line** overrides have highest priority

Example inheritance:
```
data.yaml (seq_len: 65536) 
    ↓
training.yaml (inherits data settings)
    ↓  
mamba.yaml (overrides seq_len: 1024)
    ↓
command line (overrides seq_len: 2048)  # Final value: 2048
```

## Benefits over Previous System

- **Organization**: Configuration is structured and reusable
- **Type safety**: Hydra validates configuration structure
- **Inheritance**: Share common settings across models
- **Flexibility**: Easy to override any parameter without modifying code
- **Documentation**: Configuration is self-documenting in YAML files
- **Version control**: Configuration changes are tracked in git