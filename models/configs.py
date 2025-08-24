from os import remove
import math
from pydantic import BaseModel, Field, model_validator
from typing import Union, Protocol, Optional, Literal
import torch.nn as nn
import torch
from typing_extensions import Self
"""
Central config definitions used across models.
Per-model configs that live next to their model classes are imported lazily
at the bottom (to avoid circular imports).
"""

class ModelConfig(BaseModel):
    n_ctx : int = 1024 # block_size
    vocab_size : int = 50257 
    use_adam_8_bit : bool = Field(default=False, description="""
        if true, use AdamW8bit optimizer
    """)
    is_training : bool
    eps : float = 1e-5
    nonlin : nn.Module = nn.GELU(approximate='tanh')
    class Config:
        arbitrary_types_allowed = True

class BaseTransformerConfig(ModelConfig):
    d_model : int
    d_mult : int = 4 # residual stream = d_mult*d_model
    num_heads : int
    eps : float = 1e-5
    n_layers : int
    is_causal : bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.num_heads

    class Config:
        arbitrary_types_allowed = True


class HasConfig(Protocol):
    config: Union[ModelConfig, dict]



# Import per-model config classes from their model modules here to avoid cycles
from Models.LLMS.MDLM import MDLMConfig  # noqa: E402
from Models.LLMS.D3PM import D3PMConfig  # noqa: E402
from Models.LLMS.RecurrentDepth import RecurrentDepthConfig  # noqa: E402
from Models.LLMS.SEDD import SEDDConfig  # noqa: E402
from Models.LLMS.MixtureOfExperts import MoEConfig  # noqa: E402
from Models.LLMS.MixtureOfDepths import MoDConfig  # noqa: E402
from Models.LLMS.VanillaTransformer import VanillaConfig  # noqa: E402
from Models.LLMS.Mamba import MambaConfig  # noqa: E402
from Models.LLMS.MegaByte import MegaByteConfig  # noqa: E402
from Models.LLMS.LLama3 import LLama3Config  # noqa: E402

CONFIG_MAP = {
    ModelConfig.__name__: ModelConfig,
    VanillaConfig.__name__: VanillaConfig,
    MoDConfig.__name__: MoDConfig,
    MoEConfig.__name__: MoEConfig,
    MambaConfig.__name__: MambaConfig,
    BaseTransformerConfig.__name__: BaseTransformerConfig,
    LLama3Config.__name__: LLama3Config,
    MegaByteConfig.__name__: MegaByteConfig,
    MDLMConfig.__name__: MDLMConfig,
    D3PMConfig.__name__: D3PMConfig,
    RecurrentDepthConfig.__name__: RecurrentDepthConfig,
    SEDDConfig.__name__: SEDDConfig,
}
