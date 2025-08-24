from trainer_registry import DefaultAdapter
from pydantic import BaseModel
from typing import Optional

class DefaultModelConfig(BaseModel):
    """Default model configuration that other models can inherit from"""
    vocab_size: int = 50257
    n_ctx: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    n_layers: Optional[int] = None
    
    class Config:
        arbitrary_types_allowed = True

class DefaultModelAdapter(DefaultAdapter):
    """Default model adapter that other adapters can inherit from"""
    Cfg = DefaultModelConfig