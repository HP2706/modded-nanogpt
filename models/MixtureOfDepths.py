#this file contains code inspired by: https://github.com/astramind-ai/Mixture-of-depths/tree/main
import time
from typing import Optional, Union
from numpy import dtype
from pydantic import BaseModel, Field
from sentry_sdk import is_initialized
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from beartype import beartype
from jaxtyping import jaxtyped, Int, Float, Bool
from Models.LLMS.VanillaTransformer import VanillaTransformerBlock
from modelutils import create_mask
from Models.Blocks import (
    UnEmbedding,
    MultiHeadAttention,
    MLP
)
from Models.LLMS.LLMBase import ModelMixin, TransformerMixin
from Models.LLMS.configs import BaseTransformerConfig

from Models.LLMS.LLMBase import ModelOutputMixin

class MoDOutput(ModelOutputMixin):
    ce_loss: Optional[Tensor] = None
    bce_loss: Optional[Tensor] = None
    acc: Optional[Tensor] = None
    token_pos: Optional[Tensor] = None

class TokenRouter(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.d_model, 1)
    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Float[Tensor, "batch sequence_len"]:
        return self.router(x).squeeze(-1)

class MoDBlock(nn.Module):
    def __init__(
        self, 
        cfg: "MoDConfig",
    ):
        super().__init__()
        self.cfg = cfg
        self.attn = MultiHeadAttention(cfg)
        self.mlp = MLP(cfg)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.router = TokenRouter(cfg)
        self.router_predictor = TokenRouter(cfg)
        self.capacity = cfg.capacity

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def compute_mask(
        self, 
        weights : Float[Tensor, "batch sequence_len"],
        k : int,
    )-> Bool[Tensor, "batch sequence_len"]:
        top_k_values, top_k_indices = torch.topk(weights, k, dim=1, sorted=True)
        selected_mask = torch.zeros_like(weights, dtype=torch.bool)
        selected_mask.scatter_(1, top_k_indices, True)
        return selected_mask

    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Union[
        Float[Tensor, "batch sequence_len d_model"],
        tuple[
            Float[Tensor, "batch sequence_len d_model"],
            Float[Tensor, "1"],
            Float[Tensor, "1"],
            Int[Tensor, "batch sequence_len"],
        ]
    ]:
        batch, seq_len, _ = x.shape
        k = max(1, int(self.capacity * seq_len))
        token_weights = self.router(x)
        selected_mask = self.compute_mask(token_weights, k).to(x.device).float()
        predicted_mask = self.router_predictor(x)
        if self.cfg.is_training: 
            bce_loss = F.binary_cross_entropy_with_logits(predicted_mask, selected_mask)
            rounded_predicted_mask = (predicted_mask > 0.5).float()
            acc = (rounded_predicted_mask == selected_mask).float().mean(dim=(0, 1))
        
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        
        # Optimize token selection
        flat_indices = selected_mask.reshape(-1).nonzero(as_tuple=True)[0]
        flat_x = x.reshape(-1, self.cfg.d_model)
        flat_pos_ids = position_ids.reshape(-1)

        tokens = flat_x[flat_indices].reshape(batch, k, self.cfg.d_model)
        token_pos = flat_pos_ids[flat_indices].reshape(batch, k)
        flat_indices = flat_indices.reshape(batch, k)

        # Process tokens as in classic transformer block
        attn_output = self.attn(tokens)
        x_inner = self.ln(attn_output)
        processed_tokens = self.mlp(x_inner)
        token_pos_expanded = token_pos.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)

        x = x.scatter_add(1, token_pos_expanded, processed_tokens)

        if self.cfg.is_training:
            return (
                x, 
                bce_loss, #type: ignore
                acc, #type: ignore
                token_pos
            ) 
        else:
            return x

class MoDTransformer(TransformerMixin):
    def __init__(
        self,
        cfg : "MoDConfig",
        is_master_process : bool = False
    ) -> None:
        super().__init__(
            is_master_process=is_master_process,
            cfg=cfg
        )
        self.cfg = cfg
        self.num_mod_layers = 0
        layers = []
        for i in range(cfg.n_layers):
            if (i % 2 == 0 and cfg.every_other) or not cfg.every_other:
                layers.append(MoDBlock(cfg))
                self.num_mod_layers += 1
            else:
                layers.append(VanillaTransformerBlock(cfg))
                self.num_mod_layers += 1

        self.transformer = nn.ModuleDict(dict(
            embedding = self.embedding,
            pos_embed = self.pos_embed,
            layers = nn.ModuleList(layers),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.transformer.embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        idx: Int[Tensor, "B T"], 
        targets: Optional[Int[Tensor, "B T"]] = None,
        log_metrics: bool = False
    ) -> MoDOutput:
        self.check_forward(idx, targets)
        # Allow explicit control over metric logging
        log_metrics = log_metrics or targets is not None or self.cfg.is_training

        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos_emb = self.transformer.pos_embed(pos) 
        tok_emb = self.transformer.embedding(idx) 

        x = tok_emb + pos_emb

        bce_loss_accum = torch.zeros(1, device=x.device)
        acc_accum = torch.zeros(1, device=x.device)
        token_pos_accum = torch.zeros(B, int(self.cfg.capacity*T), device=x.device)

        for i, layer in enumerate(self.transformer.layers):
            if (i % 2 == 0 or not self.cfg.every_other):
                x, bce_loss, acc, token_pos = layer(x)
                if log_metrics:
                    token_pos_accum += token_pos / self.num_mod_layers
                    bce_loss_accum += bce_loss / self.num_mod_layers 
                    acc_accum += acc / self.num_mod_layers
            else:
                x = layer(x)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        output = MoDOutput(logits=x)
        if log_metrics:
            #set the metrics
            output.bce_loss = bce_loss_accum
            output.acc = acc_accum
            output.token_pos = token_pos_accum.mean(0)

        if targets is not None:
            #set the loss
            ce_loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
            output.ce_loss = ce_loss
            output.loss = ce_loss + bce_loss_accum
        return output


class MoDConfig(BaseTransformerConfig):
    capacity: float = 0.125  # 12.5% as in mixture-of-depths paper
    every_other: bool = Field(
        default=False,
        description=(
            "if true, only every second block is a Mod and the rest is regular transformerblock"
        ),
    )
