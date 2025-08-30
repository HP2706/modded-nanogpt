# an implementation of the recurrent depth transformer 
# paper: https://www.arxiv.org/pdf/2502.05171
# github: https://github.com/seal-rg/recurrent-pretraining.git

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Int, Float

from Models.LLMS.LLMBase import TransformerMixin, ModelOutputMixin
from Models.LLMS.configs import BaseTransformerConfig
from Models.LLMS.gpt2 import Block
from trainer_registry import (
    DefaultAdapter,
    _default_group_params_for_gpt_like,
    _adam_muon_optimizers,
)


class RecurrentDepthOutput(ModelOutputMixin):
    pass


class RecurrentDepth(TransformerMixin):
    """
    Latent Recurrent-Depth Transformer (pragmatic implementation).

    Structure:
    - Prelude: l_p transformer layers to embed tokens into latent e.
    - Core (shared): l_core transformer layers applied to an adapted concat/add of (s, e), repeated r times.
    - Coda: l_c transformer layers and LM head for token logits.

    Training:
    - Samples recurrence r from a log-normal Poisson with mean `mean_recurrence`.
    - Truncated backprop through only the last k_trunc recurrent steps.
    """

    def __init__(self, cfg: "RecurrentDepthConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg

        # Prelude/Core/Coda transformer stacks
        self.prelude = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.l_p)])
        self.core = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.l_core)])
        self.coda = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.l_c)])
        self.ln_f = nn.LayerNorm(self.cfg.d_model)

        # Adapter maps concat(s, e) -> hidden or merges via addition
        if self.cfg.adapter_mode == "concat":
            self.adapter = nn.Linear(self.cfg.d_model * 2, self.cfg.d_model)
        else:
            self.adapter = nn.Identity()

        # LM head
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        if self.cfg.tie_embeddings:
            self.embedding.weight = self.lm_head.weight  # tie

        self.apply(self._init_weights)

    # ------------------------- Helpers -------------------------
    def _embed(self, idx: Int[torch.Tensor, "B T"]) -> Float[torch.Tensor, "B T D"]:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.embedding(idx) + self.pos_embed(pos)
        for blk in self.prelude:
            x = blk(x)
        return x

    def _core_once(self, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        if self.cfg.adapter_mode == "concat":
            x = torch.cat([s, e], dim=-1)
            x = self.adapter(x)
        else:  # add
            x = s + e
        for blk in self.core:
            x = blk(x)
        return x

    def _coda(self, s: torch.Tensor) -> torch.Tensor:
        x = s
        for blk in self.coda:
            x = blk(x)
        x = self.ln_f(x)
        return x

    def _sample_recurrence(self) -> int:
        # Sample r ~ 1 + Poisson(exp(N(mu, sigma^2))) with mu=log(mean)-sigma^2/2 to center the mean
        mean = max(1, int(self.cfg.mean_recurrence))
        sigma2 = float(self.cfg.lognorm_var)
        mu = torch.log(torch.tensor(float(mean), dtype=torch.float32)) - 0.5 * sigma2
        z = torch.randn((), dtype=torch.float32)
        lam = torch.exp(mu + torch.sqrt(torch.tensor(sigma2)) * z)
        # Guard against inf/nan
        lam = lam.clamp(min=1e-3, max=1e6)
        r = torch.poisson(lam).item() + 1
        r = int(max(1, min(r, self.cfg.r_max)))
        return r

    # ------------------------- Forward -------------------------
    def forward(
        self,
        idx: Int[torch.Tensor, "B T"],
        targets: Optional[Int[torch.Tensor, "B T"]] = None,
        recurrence: Optional[int] = None,
    ) -> RecurrentDepthOutput:
        self.check_forward(idx, targets)

        # Prelude embedding
        e = self._embed(idx)  # (B, T, D)

        # Initialize latent state s0 ~ N(0, s0_std^2 I)
        if self.cfg.s0_std > 0:
            s = torch.randn_like(e) * self.cfg.s0_std
        else:
            s = torch.zeros_like(e)

        # Choose recurrence
        if self.training:
            r = recurrence if recurrence is not None else self._sample_recurrence()
        else:
            r = recurrence if recurrence is not None else int(self.cfg.eval_recurrence)

        # Truncated backprop: only last k steps keep graph
        k = max(0, int(self.cfg.k_trunc))
        for i in range(1, r + 1):
            s = self._core_once(s, e)
            if self.training and i <= max(0, r - k):
                s = s.detach()

        # Coda and logits
        h = self._coda(s)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return RecurrentDepthOutput(logits=logits, loss=loss)


class RecurrentDepthConfig(BaseTransformerConfig):
    """
    Config for latent recurrent-depth transformer.

    - l_p, l_core, l_c: number of transformer layers in prelude, core (shared), and coda.
    - mean_recurrence: targeted mean recurrence tau used for sampling r.
    - lognorm_var: variance parameter for the log-normal Poisson sampling.
    - k_trunc: backprop through only the last k_trunc recurrent steps (truncate earlier ones).
    - s0_std: stddev for initial latent state s0 ~ N(0, s0_std^2 I).
    - r_max: clamp the maximum recurrence during training to keep compute bounded.
    - eval_recurrence: default recurrence at eval/inference when not specified.
    """

    l_p: int = 2
    l_core: int = 4
    l_c: int = 2

    mean_recurrence: int = 32
    lognorm_var: float = 3.0
    k_trunc: int = 8
    s0_std: float = 0.02
    r_max: int = 64
    eval_recurrence: int = 32

    # Head/weight tying
    tie_embeddings: bool = True

    # Optional choice: use adapter concat vs add
    adapter_mode: Literal["concat", "add"] = "concat"

    class Config:
        arbitrary_types_allowed = True


# ---- Adapter + Config for trainer integration ----


class RecurrentDepthAdapter(DefaultAdapter):
    Cfg = RecurrentDepthConfig

    def build(self, args, cfg: Optional[RecurrentDepthConfig]):
        cfg = cfg or RecurrentDepthConfig(
            vocab_size=50257,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            d_model=args.model_dim,
            depth_factor=2,
        )
        # Fill in any missing required fields from args
        if not hasattr(cfg, 'n_layers') or cfg.n_layers is None:
            cfg.n_layers = args.num_layers
        if not hasattr(cfg, 'n_heads') or cfg.n_heads is None:
            cfg.n_heads = args.num_heads
        if not hasattr(cfg, 'd_model') or cfg.d_model is None:
            cfg.d_model = args.model_dim
            
        model = RecurrentDepth(cfg, is_master_process=True).cuda()
        return model

    def train_step(self, model, inputs, targets, sw_num_blks, *, loss_scale, args):
        loss = model.forward(inputs.view(1, -1), targets.view(1, -1)).loss
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        return model.forward(inputs.view(1, -1), targets.view(1, -1)).loss
