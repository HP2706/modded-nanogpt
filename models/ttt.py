"""
TTT Transformer: Learning to Learn at Test-Time (https://arxiv.org/pdf/2407.04620)

This model mirrors the standard Transformer pattern but replaces attention
with a tensor-train test-time learning recurrent kernel from FLA.
It uses FLA's efficient TTT linear kernels (fused/chunk) for speed.
"""

from typing import Optional, List, cast
import warnings

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from jaxtyping import Float, Int

# Follow project import pattern used by other models
from models.configs import BaseTransformerConfig
from models.LLMBase import ModelOutputMixin, TransformerMixin
from models.components.Blocks import MLP

# Efficient TTT linear kernels from FLA
from fla.ops.ttt import fused_chunk_ttt_linear, chunk_ttt_linear

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
    except ImportError:
        selective_state_update, mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined = None, None, None
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_update, causal_conv1d_fn = None, None
    is_fast_path_available = selective_state_update is not None

class TTTOutput(ModelOutputMixin):
    pass

class TTTLinearAttention(nn.Module):
    """
    Multi-head projection + FLA TTT linear kernel.

    Shapes:
      - input x: [B, T, C]
      - projects to q, k, v of shape [B, H, T, D]
      - calls FLA kernel, returns [B, H, T, D], then projects back to [B, T, C]
    """

    def __init__(self, cfg: BaseTransformerConfig, *, use_fused: bool = True, chunk_size: int = 16, eta: float = 5e-3,
                 learn_init_state: bool = True):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.cfg = cfg
        self.n_head = cfg.num_heads
        self.d_head = cfg.d_head
        self.use_fused = use_fused
        self.chunk_size = int(chunk_size)
        # Learning rate for hidden state updates in TTT kernel
        self.eta_base = float(eta)

        # Per-token learnable LR gate η(x) = η_base * sigmoid(W_lr x) -> per-head
        self.lr_gate = nn.Linear(cfg.d_model, self.n_head, bias=False)

        # qkv and output projections
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

        # Per-head norm parameters used inside FLA kernel (GroupNorm + residual)
        self.w = nn.Parameter(torch.ones(self.n_head, self.d_head))
        self.b = nn.Parameter(torch.zeros(self.n_head, self.d_head))

        # Optional learnable initial state (W0 alias): shapes follow FLA API
        self.learn_init_state = learn_init_state
        if self.learn_init_state:
            # h0: [1, H, D, D], hb0: [1, H, 1, D]
            h0 = torch.zeros(1, self.n_head, self.d_head, self.d_head)
            hb0 = torch.zeros(1, self.n_head, 1, self.d_head)
            # small random init to break symmetry
            nn.init.normal_(h0, mean=0.0, std=1e-3)
            nn.init.zeros_(hb0)
            self.h0 = nn.Parameter(h0)
            self.hb0 = nn.Parameter(hb0)
        else:
            self.h0 = None
            self.hb0 = None

    def forward(self, x: Float[Tensor, "batch sequence_len d_model"]) -> Float[Tensor, "batch sequence_len d_model"]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # Shape to [B, H, T, D]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Build per-token, per-head eta from the input tokens x before projection
        # gate: [B, T, H] -> eta: [B, H, T, 1]
        gate = torch.sigmoid(self.lr_gate(x))
        eta = self.eta_base * gate.permute(0, 2, 1).unsqueeze(-1)

        # Call FLA kernel. Note: the FLA API expects [B, H, T, D].
        # Fused kernel preferred; fallback to chunk kernel if needed.
        scale = self.d_head ** -0.5
        init_h = self.h0.expand(B, -1, -1, -1) if self.h0 is not None else None
        init_hb = self.hb0.expand(B, -1, -1, -1) if self.hb0 is not None else None
        if self.use_fused:
            o, _, _ = fused_chunk_ttt_linear(
                q, k, v, self.w, self.b, eta, scale=scale, eps=self.cfg.eps, chunk_size=self.chunk_size,
                initial_state=init_h, initial_state_bias=init_hb, output_final_state=False, cu_seqlens=None,
            )
        else:
            o, _, _ = chunk_ttt_linear(
                q, k, v, self.w, self.b, eta, scale=scale, eps=self.cfg.eps, chunk_size=self.chunk_size,
                initial_state=init_h, initial_state_bias=init_hb, output_final_state=False, cu_seqlens=None,
            )

        # Back to [B, T, C]
        o = o.transpose(1, 2).contiguous().view(B, T, C)
        o = self.c_proj(o)
        return o


class TTTBlock(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig, *, use_fused: bool, chunk_size: int, eta: float,
                 learn_init_state: bool, use_mamba_backbone: bool, d_conv: int):
        super().__init__()
        self.use_mamba_backbone = use_mamba_backbone
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        # Optional temporal conv (Mamba-style backbone)
        if self.use_mamba_backbone:
            self.temporal_conv = CausalDepthwiseConv1d(cfg.d_model, max(1, int(d_conv)))
        else:
            self.temporal_conv = None
        self.attn = TTTLinearAttention(
            cfg, use_fused=use_fused, chunk_size=chunk_size, eta=eta, learn_init_state=learn_init_state
        )
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: Float[Tensor, "batch sequence_len d_model"]) -> Float[Tensor, "batch sequence_len d_model"]:
        y = self.ln_1(x)
        if self.temporal_conv is not None:
            y = self.temporal_conv(y)
        x = x + self.attn(y)
        x = x + self.mlp(self.ln_2(x))
        return x


class TTT(TransformerMixin):
    def __init__(self, cfg: "TTTConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg

        self.blocks = nn.ModuleDict(
            dict(
                embedding=self.embedding,
                pos_embed=self.pos_embed,
                layers=nn.ModuleList(
                    [
                        TTTBlock(
                            cfg,
                            use_fused=cfg.use_fused_ttt,
                            chunk_size=cfg.chunk_size,
                            eta=cfg.eta,
                            learn_init_state=cfg.learn_init_state,
                            use_mamba_backbone=cfg.use_mamba_backbone,
                            d_conv=cfg.d_conv,
                        )
                        for _ in range(cfg.n_layers)
                    ]
                ),
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
        sw_num_blks: Optional[Tensor] = None,  # kept for trainer API compatibility
    ) -> TTTOutput:
        self.check_forward(idx, targets)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.blocks.pos_embed(pos)  # type: ignore[attr-defined]
        tok_emb = self.blocks.embedding(idx)  # type: ignore[attr-defined]
        x = tok_emb + pos_emb

        for block in cast(List[TTTBlock], self.blocks.layers):  # type: ignore[attr-defined]
            x = block(x)

        x = self.blocks.ln_f(x)  # type: ignore[attr-defined]
        logits = self.lm_head(x)

        if targets is None:
            return TTTOutput(logits=logits)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return TTTOutput(logits=logits, loss=loss)


class TTTConfig(BaseTransformerConfig):
    """
    Config additions for TTT:
      - use_fused_ttt: prefer fused kernel implementation
      - chunk_size: chunk length for kernels
      - eta: learning rate for hidden state updates (TTT rule)
    """

    use_fused_ttt: bool = True
    chunk_size: int = 16
    eta: float = 5e-3
    learn_init_state: bool = True
    # Backbone options
    use_mamba_backbone: bool = False
    d_conv: int = 4


# ---- Adapter for trainer integration ----
from typing import Any
from pydantic import BaseModel
from trainer_registry import DefaultAdapter


class TTTCfg(BaseModel):
    vocab_size: int = 50257
    n_ctx: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    n_layers: Optional[int] = None
    use_fused_ttt: bool = True
    chunk_size: int = 16
    eta: float = 5e-3
    learn_init_state: bool = True
    use_mamba_backbone: bool = False
    d_conv: int = 4

    class Config:
        arbitrary_types_allowed = True


class TTTAdapter(DefaultAdapter):
    Cfg = TTTCfg

    def build(self, args: Any, cfg: Optional[TTTCfg]):
        cfg = cfg or TTTCfg()
        mcfg = TTTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.n_ctx or args.seq_len,
            d_model=cfg.d_model or args.model_dim,
            num_heads=cfg.num_heads or args.num_heads,
            n_layers=cfg.n_layers or args.num_layers,
            is_training=True,
            use_fused_ttt=cfg.use_fused_ttt,
            chunk_size=cfg.chunk_size,
            eta=cfg.eta,
            learn_init_state=cfg.learn_init_state,
            use_mamba_backbone=cfg.use_mamba_backbone,
            d_conv=cfg.d_conv,
        )
        model = TTT(mcfg, is_master_process=True).cuda()
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
