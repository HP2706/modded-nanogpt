import time
from typing import List, Optional, Union, cast
import torch
from jaxtyping import jaxtyped, Int, Float, Bool
from beartype import beartype
from torch import nn, Tensor
from torch.nn import functional as F
from Models.LLMS.configs import BaseTransformerConfig
from Models.Blocks import (
    UnEmbedding,
    MultiHeadAttention,
    MLP
)
from Models.LLMS.gpt2 import Block
from Models.LLMS.LLMBase import ModelOutputMixin, TransformerMixin
from jaxtyping import Float, Int

class VanillaTransformerOutput(ModelOutputMixin):
    pass

class VanillaTransformerBlock(nn.Module):
    def __init__(self, cfg : BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Float[Tensor, "batch sequence_len d_model"]:
        x = x + self.attn(self.ln(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VanillaTransformer(TransformerMixin):
    def __init__(
        self, 
        cfg : "VanillaConfig", 
        is_master_process : bool = True
    ):
        super().__init__(
            is_master_process=is_master_process,
            cfg=cfg
        )
        self.transformer = nn.ModuleDict(dict(
            embedding = self.embedding,
            pos_embed = self.pos_embed,
            layers = nn.ModuleList([VanillaTransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.transformer.embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        idx : Int[torch.Tensor, "B T D"], 
        targets : Optional[Int[torch.Tensor, "B T D"]] = None
    ) -> VanillaTransformerOutput:
        self.check_forward(idx, targets)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.pos_embed(pos) # position embeddings of shape (T, d_model)
        tok_emb = self.transformer.embedding(idx) # token embeddings of shape (B, T, d_model)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in cast(List[VanillaTransformerBlock], self.transformer.layers):
            x = block.forward(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        if targets is None:
            return VanillaTransformerOutput(logits=x)
        else:
            return VanillaTransformerOutput(
                logits=x, 
                loss=F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
            )


class VanillaConfig(BaseTransformerConfig):
    pass
   
