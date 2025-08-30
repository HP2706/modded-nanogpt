"""
Titans Transformer: online learned linear recurrent kernel per Titans update rule.

Follows the same pattern as our other models (LN → Attention → LN → MLP),
with attention implemented via FLA's Titans chunked kernel.
"""

from typing import Optional, List, cast, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int

from models.configs import BaseTransformerConfig
from models.LLMBase import ModelOutputMixin, TransformerMixin
from models.components.Blocks import MLP

# Titans kernel (chunked implementation)
from fla.ops.titans import chunk_titans_linear


class TitansOutput(ModelOutputMixin):
    pass


class TitansLinearAttention(nn.Module):
    """
    Multi-head projection + FLA Titans linear kernel.

    - Input x: [B, T, C]
    - q, k, v: [B, H, T, D], with q, k L2-normalized along D
    - Per-token, per-head scalars theta, alpha, eta: [B, H, T, 1], produced from x
    - FLA kernel returns o: [B, H, T, D]
    - Project back to [B, T, C]
    """

    def __init__(self, cfg: BaseTransformerConfig, *, chunk_size: int = 16):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.cfg = cfg
        self.n_head = cfg.num_heads
        self.d_head = cfg.d_head
        self.chunk_size = int(chunk_size)

        # qkv and output projections
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

        # Project to per-head Titans parameters (theta, alpha, eta) per token
        # shape after projection: [B, T, 3*H] → reshape/split to [B, H, T, 1]
        self.param_proj = nn.Linear(cfg.d_model, 3 * self.n_head)

        # Per-head norm parameters inside kernel
        self.w = nn.Parameter(torch.ones(self.n_head, self.d_head))
        self.b = nn.Parameter(torch.zeros(self.n_head, self.d_head))

    def forward(self, x: Float[Tensor, "batch sequence_len d_model"]) -> Float[Tensor, "batch sequence_len d_model"]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # [B, H, T, D]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # L2-normalize q, k along head dimension (last dim)
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        # Titans parameters per token per head, constrained to (0, 1) via sigmoid
        params = self.param_proj(x)  # [B, T, 3H]
        params = params.view(B, T, self.n_head, 3).permute(0, 2, 1, 3)  # [B, H, T, 3]
        theta, alpha, eta = params.unbind(dim=-1)
        theta = torch.sigmoid(theta).unsqueeze(-1)  # [B, H, T, 1]
        alpha = torch.sigmoid(alpha).unsqueeze(-1)
        eta = torch.sigmoid(eta).unsqueeze(-1)

        # Run Titans kernel
        o, _ = chunk_titans_linear(
            q, k, v, self.w, self.b, theta, alpha, eta,
            self.cfg.eps, self.chunk_size, None, False
        )

        # Back to [B, T, C]
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        o = self.c_proj(o)
        return o


class TitansBlock(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig, *, chunk_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = TitansLinearAttention(cfg, chunk_size=chunk_size)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Float[Tensor, "batch sequence_len d_model"]) -> Float[Tensor, "batch sequence_len d_model"]:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Titans(TransformerMixin):
    def __init__(self, cfg: "TitansConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg

        self.blocks = nn.ModuleDict(
            dict(
                embedding=self.embedding,
                pos_embed=self.pos_embed,
                layers=nn.ModuleList([
                    TitansBlock(cfg, chunk_size=cfg.chunk_size)
                    for _ in range(cfg.n_layers)
                ]),
                ln_f=nn.LayerNorm(cfg.d_model),
            )
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.blocks["embedding"].weight = self.lm_head.weight  # type: ignore[index]

        self.apply(self._init_weights)

    def forward(
        self,
        idx: Int[Tensor, "B T"],
        targets: Optional[Int[Tensor, "B T"]] = None,
        sw_num_blks: Optional[Tensor] = None,
    ) -> TitansOutput:
        self.check_forward(idx, targets)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.blocks.pos_embed(pos)  # type: ignore[attr-defined]
        tok_emb = self.blocks.embedding(idx)  # type: ignore[attr-defined]
        x = tok_emb + pos_emb

        for block in cast(List[TitansBlock], self.blocks.layers):  # type: ignore[attr-defined]
            x = block(x)

        x = self.blocks.ln_f(x)  # type: ignore[attr-defined]
        logits = self.lm_head(x)
        if targets is None:
            return TitansOutput(logits=logits)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return TitansOutput(logits=logits, loss=loss)


class TitansConfig(BaseTransformerConfig):
    chunk_size: int = 16


# ---- Adapter for trainer integration ----
from pydantic import BaseModel
from trainer_registry import DefaultAdapter


class TitansCfg(BaseModel):
    vocab_size: int = 50257
    n_ctx: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    n_layers: Optional[int] = None
    chunk_size: int = 16

    class Config:
        arbitrary_types_allowed = True


class TitansAdapter(DefaultAdapter):
    Cfg = TitansCfg

    def build(self, args: Any, cfg: Optional[TitansCfg]):
        cfg = cfg or TitansCfg()
        mcfg = TitansConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.n_ctx or args.seq_len,
            d_model=cfg.d_model or args.model_dim,
            num_heads=cfg.num_heads or args.num_heads,
            n_layers=cfg.n_layers or args.num_layers,
            is_training=True,
            chunk_size=cfg.chunk_size,
        )
        model = Titans(mcfg, is_master_process=True).cuda()
        return model

    def train_step(self, model, inputs, targets, sw_num_blks, *, loss_scale, args):
        out = model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long))
        loss = out.loss
        assert loss is not None
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        out = model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long))
        assert out.loss is not None
        return out.loss

