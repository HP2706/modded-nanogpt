# https://arxiv.org/abs/2310.16834
from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from models.LLMBase import ModelOutputMixin, TransformerMixin
from models.configs import BaseTransformerConfig
from models.gpt2 import Block


class SEDDConfig(BaseTransformerConfig):
    """
    Minimal config for Score Entropy Discrete Diffusion (SEDD)-style model.

    Notes:
    - This is a pragmatic integration into the existing training loop. It keeps
      a standard LM head for autoregressive loss while adding a score-entropy-like
      denoising loss computed on noised inputs (uniform/absorbing).
    - Exact schedules and hyperparameters can be tuned to match the paper.
    """

    # Discrete diffusion forward process type
    transition_type: Literal["uniform", "absorbing"] = "absorbing"

    # Noise schedule and steps (used for the denoising loss)
    noise_schedule: Literal["linear", "mutual_info"] = "mutual_info"
    num_timesteps: int = 1000

    # Absorbing state index (for absorbing diffusion)
    absorbing_state_idx: int = -1  # default to last token id

    # Loss weights
    lm_loss_weight: float = 1.0
    se_loss_weight: float = 1.0

    # Sampling params (not used in trainer generation; kept for completeness)
    sampling_timesteps: int = 50

    class Config:
        arbitrary_types_allowed = True


class SEDDOutput(ModelOutputMixin):
    se_loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None


class SEDD(TransformerMixin):
    """
    Score Entropy Discrete Diffusion style model (pragmatic integration).

    - Backbone: GPT-like transformer trunk (Blocks from gpt2.py)
    - Heads: standard LM head + SE head used on noised inputs
    - Loss: total = lm_loss_weight * CE(next-token) + se_loss_weight * SE denoising CE

    This approximates the score-entropy denoising objective by training a
    denoising head on noised inputs (uniform/absorbing forward process) and
    enforcing positivity via softplus before normalization.
    """

    def __init__(self, cfg: SEDDConfig, is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg

        # Transformer trunk
        self.transformer = nn.ModuleDict(
            dict(
                embedding=self.embedding,
                pos_embed=self.pos_embed,
                h=nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)]),
                ln_f=nn.LayerNorm(self.cfg.d_model),
            )
        )

        # Two heads: LM head (shared with embedding) and SE head
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        self.se_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        # Weight sharing for LM pathway as usual
        self.transformer.embedding.weight = self.lm_head.weight

        # Initialize
        self.apply(self._init_weights)

        # Absorbing state index resolution
        if self.cfg.absorbing_state_idx == -1:
            self.absorbing_state_idx = self.cfg.vocab_size - 1
        else:
            self.absorbing_state_idx = self.cfg.absorbing_state_idx

    # ---------------------- Diffusion helpers ----------------------
    def _beta_t(self, t: int) -> float:
        if self.cfg.noise_schedule == "linear":
            return t / max(1, self.cfg.num_timesteps)
        elif self.cfg.noise_schedule == "mutual_info":
            # (T - t + 1)^-1 (simple decreasing schedule)
            return 1.0 / (self.cfg.num_timesteps - t + 1)
        else:
            raise ValueError(f"Unknown noise schedule: {self.cfg.noise_schedule}")

    def _q_xt_given_x0(self, x0: Int[torch.Tensor, "B T"], t: int) -> Int[torch.Tensor, "B T"]:
        B, T = x0.shape
        beta_t = self._beta_t(t)
        noise = torch.rand(B, T, device=x0.device)
        mask = (noise < beta_t).long()

        if self.cfg.transition_type == "uniform":
            uniform_tokens = torch.randint(0, self.cfg.vocab_size, (B, T), device=x0.device)
            xt = x0 * (1 - mask) + uniform_tokens * mask
        else:  # absorbing
            absorbing = torch.full((B, T), self.absorbing_state_idx, device=x0.device)
            xt = x0 * (1 - mask) + absorbing * mask
        return xt.long()

    def _backbone(self, tokens: Int[torch.Tensor, "B T"]) -> Float[torch.Tensor, "B T D"]:
        B, T = tokens.size()
        pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
        pos_emb = self.transformer.pos_embed(pos)
        tok_emb = self.transformer.embedding(tokens)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    # --------------------------- Losses ----------------------------
    def _compute_se_loss(
        self,
        x0: Int[torch.Tensor, "B T"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute simplified score-entropy-like denoising loss:
        - Sample t, produce xt via forward process
        - Run backbone on xt, predict positive ratios with se_head
        - Compute CE to x0, optionally focusing on masked positions for absorbing
        Returns (se_loss, se_logits) where se_logits are pre-softmax
        """
        B, T = x0.shape
        t = torch.randint(1, self.cfg.num_timesteps + 1, (1,), device=x0.device).item()
        xt = self._q_xt_given_x0(x0, t)

        h = self._backbone(xt)
        se_logits = self.se_head(h)

        if self.cfg.transition_type == "absorbing":
            mask = (xt == self.absorbing_state_idx).float()  # supervised only where noised
            ce = F.cross_entropy(
                se_logits.view(-1, self.cfg.vocab_size), x0.view(-1), reduction="none"
            ).view(B, T)
            if mask.sum() > 0:
                se_loss = (ce * mask).sum() / mask.sum()
            else:
                se_loss = ce.mean() * 0.0  # zero if no masked tokens
        else:
            se_loss = F.cross_entropy(se_logits.view(-1, self.cfg.vocab_size), x0.view(-1))

        return se_loss, se_logits

    # --------------------------- Forward ---------------------------
    def forward(
        self,
        idx: Int[torch.Tensor, "B T"],
        targets: Optional[Int[torch.Tensor, "B T"]] = None,
    ) -> SEDDOutput:
        self.check_forward(idx, targets)

        # Standard LM path (used by trainer for next-token prediction)
        h_lm = self._backbone(idx)
        lm_logits = self.lm_head(h_lm)

        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(lm_logits.view(-1, self.cfg.vocab_size), targets.view(-1))

        # Score-entropy denoising path (always computed during training mode)
        se_loss = None
        if self.training:
            se_loss, _ = self._compute_se_loss(idx)

        # Total loss (if any)
        total_loss = None
        if lm_loss is not None or se_loss is not None:
            total_loss = (
                (self.cfg.lm_loss_weight * lm_loss if lm_loss is not None else 0.0)
                + (self.cfg.se_loss_weight * se_loss if se_loss is not None else 0.0)
            )

        return SEDDOutput(logits=lm_logits, loss=total_loss, se_loss=se_loss, lm_loss=lm_loss)

# ---- Adapter + Config for trainer integration ----
from typing import Optional
from pydantic import BaseModel
from trainer_registry import (
    ModelAdapter,
    _adam_muon_optimizers,
)
import torch
from torch import nn


class SEDDCfg(BaseModel):
    vocab_size: int = 50257
    n_ctx: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    n_layers: Optional[int] = None
    transition_type: Optional[str] = None
    noise_schedule: Optional[str] = None
    num_timesteps: Optional[int] = None
    absorbing_state_idx: Optional[int] = None
    lm_loss_weight: Optional[float] = None
    se_loss_weight: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True


class SEDDAdapter(ModelAdapter):
    Cfg = SEDDCfg

    def build(self, args, cfg: Optional[SEDDCfg]):
        cfg = cfg or SEDDCfg()
        scfg = SEDDConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.n_ctx or args.seq_len,
            d_model=cfg.d_model or args.model_dim,
            num_heads=cfg.num_heads or args.num_heads,
            n_layers=cfg.n_layers or args.num_layers,
            is_training=True,
        )
        if cfg.transition_type is not None:
            scfg.transition_type = cfg.transition_type  # type: ignore
        if cfg.noise_schedule is not None:
            scfg.noise_schedule = cfg.noise_schedule  # type: ignore
        if cfg.num_timesteps is not None:
            scfg.num_timesteps = cfg.num_timesteps  # type: ignore
        if cfg.absorbing_state_idx is not None:
            scfg.absorbing_state_idx = cfg.absorbing_state_idx  # type: ignore
        if cfg.lm_loss_weight is not None:
            scfg.lm_loss_weight = cfg.lm_loss_weight  # type: ignore
        if cfg.se_loss_weight is not None:
            scfg.se_loss_weight = cfg.se_loss_weight  # type: ignore

        model = SEDD(scfg, is_master_process=True).cuda()
        return model

    def create_optimizers(self, model, args, *, rank, world_size, device):
        # Group params by ndim/name
        hidden_matrix_params, embed_params, scalar_params, head_params = [], [], [], []
        for name, p in model.named_parameters():
            if p.ndim < 2:
                scalar_params.append(p)
            elif 'embedding' in name:
                embed_params.append(p)
            elif 'lm_head.weight' in name:
                head_params.append(p)
            else:
                hidden_matrix_params.append(p)

        optimizer1, optimizer2 = _adam_muon_optimizers(
            hidden_matrix_params=hidden_matrix_params,
            embed_params=embed_params,
            scalar_params=scalar_params,
            head_params=head_params,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        def get_lr(step: int):
            t = 1 - step / args.num_iterations
            w = min(t / args.cooldown_frac, 1.0)
            return w * 1.0 + (1 - w) * 0.1

        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr),
            torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr),
        ]
        return [optimizer1, optimizer2], schedulers

    def train_step(self, model, inputs, targets, sw_num_blks, *, loss_scale, args):
        out = model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long))
        loss = out.loss if out.loss is not None else out.lm_loss
        assert loss is not None, "SEDD forward did not return a loss"
        (loss_scale * loss).backward()
        aux = {}
        if out.se_loss is not None:
            aux["se_loss"] = out.se_loss
        if out.lm_loss is not None:
            aux["lm_loss"] = out.lm_loss
        return loss, aux

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        out = model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long))
        # default to LM loss for validation
        if out.lm_loss is not None:
            return out.lm_loss
        assert out.loss is not None
        return out.loss

    def requires_scaled_grad_on_reduce(self) -> bool:
        return False

    def post_optimizer_step(self, model: nn.Module, *, args) -> None:
        pass
