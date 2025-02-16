import wandb
from typing import Union
import torch.nn as nn

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

def numel_params_million(model : nn.Module):
    return sum(p.numel() for p in model.parameters()) / 1e6

def unwrap(model):
    """Unwrap a torch.compiled model to get the original model class"""
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model