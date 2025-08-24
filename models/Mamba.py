# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
from Models.LLMS.LLMBase import ModelMixin
from utils import LRConfig, get_device
from Models.LLMS.LLMBase import ModelOutputMixin
from Models.Blocks import GatedMLP
from jaxtyping import Float, Int
import torch
import torch.nn as nn


if get_device() == "cuda":
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as MambaMixer
        from mamba_ssm.modules.mamba2 import Mamba2 as Mamba2Mixer
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from dataclasses import dataclass
from pydantic import BaseModel, Field
from Models.LLMS.configs import ModelConfig
from typing import Optional, Tuple, Literal
import torch
from torch import nn, Tensor



class MambaBlock(nn.Module):
    def __init__(
        self, 
        cfg : MambaConfig,
        layer_idx : int,
    ):
        super().__init__()
        self.residual_in_fp32 = cfg.residual_in_fp32
        self.fused_add_norm = cfg.fused_add_norm
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = MambaMixer(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=layer_idx
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, cfg.d_model)

    def forward(
        self, 
        hidden_states: Float[Tensor, "batch_size x seq_len x d_model"], 
        residual: Optional[Float[Tensor, "batch_size x seq_len x d_model"]] = None
    ) -> Tuple[
            Float[Tensor, "batch_size x seq_len x d_model"], 
            Float[Tensor, "batch_size x seq_len x d_model"]
        ]:
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class MambaConfig(ModelConfig):
    d_model: int
    d_state: int
    n_layers: int
    pad_token_id: int = Field(default=0, description="Padding token id.")
    bos_token_id: int = Field(default=0, description="The id of the beginning of sentence token in the vocabulary.")
    fused_add_norm: bool = Field(default=False, description="Whether or not to fuse add and norm")
    eos_token_id: int = Field(default=0, description="The id of the end of sentence token in the vocabulary.")
    expand: int = Field(default=2, description="Expanding factor used to determine the intermediate size.")
    d_conv: int = Field(default=4, description="Size of the convolution kernel.")
    use_bias: bool = Field(default=False, description="Whether or not to use bias in ['in_proj', 'out_proj'] of the mixer block")
    use_conv_bias: bool = Field(default=True, description="Whether or not to use bias in the convolution layer of the mixer block.")
    hidden_act: str = Field(default="silu", description="The non-linear activation function in the decoder.")
    initializer_range: float = Field(default=0.1, description="Stddev of initializer for weight matrices.")
    residual_in_fp32: bool = Field(default=True, description="Whether or not residuals should be in float32.")
    time_step_rank: int | Literal["auto"] = Field(default="auto", description="Rank of the discretization projection matrix.")
    time_step_scale: float = Field(default=1.0, description="Scale used to scale dt_proj.bias.")
    time_step_min: float = Field(default=0.001, description="Minimum time_step used to bound dt_proj.bias.")
    time_step_max: float = Field(default=0.1, description="Maximum time_step used to bound dt_proj.bias.")
    time_step_init_scheme: Literal["random", "uniform"] = Field(default="random", description="Init scheme for dt_proj.weight.")
    time_step_floor: float = Field(default=1e-4, description="Minimum clamping value for dt_proj.bias initialization.")
    rescale_prenorm_residual: bool = Field(default=False, description="Whether to rescale out_proj weights when initializing.")

    @property
    def d_intermediate(self) -> int:
        return int(self.expand * self.d_model)

    @property
    def time_step_rank_value(self) -> int:
        return math.ceil(self.d_model / 16) if self.time_step_rank == "auto" else self.time_step_rank

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.time_step_rank == "auto":
            self.time_step_rank = math.ceil(self.d_model / 16)


class Mamba(ModelMixin):
    def __init__(
        self,
        config: MambaConfig,
        is_master_process: bool = True,
    ) -> None:
        super().__init__(cfg=config, is_master_process=is_master_process)
        self.cfg = config
        self.config = config
        self.is_master_process = True
        self.layers = nn.ModuleList([MambaBlock(self.cfg, i) for i in range(self.cfg.n_layers)])
        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.norm_f = RMSNorm(self.cfg.d_model)

        self.apply(
            partial(
                _init_weights,
                n_layer=self.cfg.n_layers
            )
        )
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self, **kwargs):
        _init_weights(**kwargs)

    def forward(
        self, 
        input_ids: Int[Tensor, "batch_size x seq_len"], 
        position_ids: Optional[Int[Tensor, "batch_size x seq_len"]] = None
    )-> ModelOutputMixin:
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual
            )
        if not self.cfg.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.cfg.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        lm_logits = self.lm_head(hidden_states)
        if position_ids is not None:
            ce_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), position_ids.view(-1))
            return ModelOutputMixin(logits=lm_logits, loss=ce_loss)
        return ModelOutputMixin(logits=lm_logits)
