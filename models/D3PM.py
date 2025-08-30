import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int

from models.LLMBase import TransformerMixin, ModelOutputMixin
from models.configs import BaseTransformerConfig
from models.gpt2 import Block
from trainer_registry import (
    DefaultAdapter,
    _default_group_params_for_gpt_like,
    _adam_muon_optimizers,
)

class D3PMOutput(ModelOutputMixin):
    logits: Tensor
    loss: Optional[Tensor] = None
    diffusion_loss: Optional[Tensor] = None
    auxiliary_loss: Optional[Tensor] = None

class D3PM(TransformerMixin):
    def __init__(self, cfg: "D3PMConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        
        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            embedding=self.embedding,
            pos_embed=self.pos_embed,
            h=nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)]),
            ln_f=nn.LayerNorm(self.cfg.d_model),
        ))
        
        # Output head for denoising (predicts logits for x0 given xt and t)
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Set absorbing state index
        if self.cfg.absorbing_state_idx == -1:
            self.absorbing_state_idx = self.cfg.vocab_size - 1  # Last token is [MASK]
        else:
            self.absorbing_state_idx = self.cfg.absorbing_state_idx
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.cfg.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def beta_t(self, t: int) -> float:
        """Compute beta_t based on the noise schedule."""
        if self.cfg.noise_schedule == "cosine":
            s = 0.008
            return 1 - math.cos((t / self.cfg.num_timesteps + s) / (1 + s) * math.pi / 2) / \
                         math.cos((t / self.cfg.num_timesteps + s) / (1 + s) * math.pi / 2)
        elif self.cfg.noise_schedule == "linear":
            if self.cfg.transition_type == "gaussian":
                # Linear schedule for Gaussian diffusion
                return self.cfg.gaussian_beta_min + (t / self.cfg.num_timesteps) * \
                       (self.cfg.gaussian_beta_max - self.cfg.gaussian_beta_min)
            else:
                # Linear schedule for other types
                return t / self.cfg.num_timesteps
        elif self.cfg.noise_schedule == "mutual_info":
            # For absorbing state, this reduces to (T-t+1)^-1
            return 1.0 / (self.cfg.num_timesteps - t + 1)
        else:
            raise ValueError(f"Unknown noise schedule: {self.cfg.noise_schedule}")
    
    def q_xt_given_x0(self, x0: Int[Tensor, "batch seq"], t: int) -> Int[Tensor, "batch seq"]:
        """Sample xt from q(xt|x0) according to the forward process."""
        batch_size, seq_len = x0.shape
        
        # Compute beta_t
        beta_t = self.beta_t(t)
        
        # Sample noise
        noise = torch.rand(batch_size, seq_len, device=x0.device)
        
        if self.cfg.transition_type == "uniform":
            # Uniform transition: with probability beta_t, transition to any token uniformly
            mask = (noise < beta_t).long()
            uniform_tokens = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len), device=x0.device)
            xt = x0 * (1 - mask) + uniform_tokens * mask
        elif self.cfg.transition_type == "absorbing":
            # Absorbing state: with probability beta_t, transition to absorbing state
            mask = (noise < beta_t).long()
            absorbing_tokens = torch.full((batch_size, seq_len), self.absorbing_state_idx, device=x0.device)
            xt = x0 * (1 - mask) + absorbing_tokens * mask
        elif self.cfg.transition_type == "gaussian":
            # Gaussian-like transition: transition to nearby tokens with higher probability
            # This is a simplified implementation
            mask = (noise < beta_t).long()
            # Sample from a distribution that favors nearby tokens
            offsets = torch.randint(-3, 4, (batch_size, seq_len), device=x0.device)
            gaussian_tokens = torch.clamp(x0 + offsets, 0, self.cfg.vocab_size - 1)
            xt = x0 * (1 - mask) + gaussian_tokens * mask
        else:
            # Default to absorbing
            mask = (noise < beta_t).long()
            absorbing_tokens = torch.full((batch_size, seq_len), self.absorbing_state_idx, device=x0.device)
            xt = x0 * (1 - mask) + absorbing_tokens * mask
            
        return xt.long()
    
    def denoise(self, xt: Int[Tensor, "batch seq"], t: int) -> Float[Tensor, "batch seq vocab"]:
        """Denoise xt at time t to predict x0 logits."""
        B, T = xt.size()
        
        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=xt.device)
        pos_emb = self.transformer.pos_embed(pos)
        tok_emb = self.transformer.embedding(xt)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Output logits for x0
        logits = self.lm_head(x)
        
        return logits
    
    def compute_losses(
        self, 
        x0: Int[Tensor, "batch seq"], 
        t: Optional[int] = None
    ) -> Tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """Compute the diffusion loss and auxiliary loss."""
        batch_size, seq_len = x0.shape
        
        # Sample random time step if not provided
        if t is None:
            t = torch.randint(1, self.cfg.num_timesteps + 1, (1,)).item()
        
        # Forward process: get xt from x0
        xt = self.q_xt_given_x0(x0, t)
        
        # Denoise: predict x0 from xt and t
        pred_logits = self.denoise(xt, t)
        
        # Compute auxiliary loss (cross-entropy between predicted and true x0)
        auxiliary_loss = F.cross_entropy(
            pred_logits.view(-1, self.cfg.vocab_size), 
            x0.view(-1), 
            reduction='none'
        ).view(batch_size, seq_len).mean()
        
        # Compute diffusion loss (KL divergence terms)
        # For simplicity, we approximate this with weighted cross-entropy
        # In practice, this would involve computing the full KL divergence
        beta_t = self.beta_t(t)
        
        if self.cfg.transition_type == "absorbing":
            # For absorbing state, the loss is computed only on masked tokens
            mask = (xt == self.absorbing_state_idx).float()
            if mask.sum() > 0:
                ce_loss = F.cross_entropy(
                    pred_logits.view(-1, self.cfg.vocab_size), 
                    x0.view(-1), 
                    reduction='none'
                ).view(batch_size, seq_len)
                diffusion_loss = (ce_loss * mask).sum() / mask.sum()
            else:
                diffusion_loss = torch.tensor(0.0, device=x0.device)
        else:
            # For other transition types, use standard cross-entropy
            diffusion_loss = F.cross_entropy(pred_logits.view(-1, self.cfg.vocab_size), x0.view(-1))
        
        return diffusion_loss, auxiliary_loss, pred_logits
    
    def forward(
        self, 
        idx: Int[Tensor, "batch seq"], 
        targets: Optional[Int[Tensor, "batch seq"]] = None,
        compute_diffusion_loss: bool = True
    ) -> D3PMOutput:
        """Forward pass for D3PM."""
        self.check_forward(idx, targets)
        
        B, T = idx.size()
        
        # Standard language modeling loss (if targets provided)
        lm_loss = None
        if targets is not None:
            # Use the model as a standard language model
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.pos_embed(pos)
            tok_emb = self.transformer.embedding(idx)
            x = tok_emb + pos_emb
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Denoising forward pass
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.pos_embed(pos)
            tok_emb = self.transformer.embedding(idx)
            x = tok_emb + pos_emb
            
            for block in self.transformer.h:
                x = block(x)
            
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
        
        # Compute diffusion losses
        diffusion_loss = None
        auxiliary_loss = None
        if compute_diffusion_loss:
            diffusion_loss, auxiliary_loss, _ = self.compute_losses(idx)
        
        # Total loss
        loss = None
        if lm_loss is not None:
            loss = lm_loss
        elif diffusion_loss is not None and auxiliary_loss is not None:
            # Hybrid loss as in the paper
            loss = diffusion_loss + self.cfg.auxiliary_loss_weight * auxiliary_loss
        elif diffusion_loss is not None:
            loss = diffusion_loss
        
        return D3PMOutput(
            logits=logits, 
            loss=loss, 
            diffusion_loss=diffusion_loss,
            auxiliary_loss=auxiliary_loss
        )
    
    def sample(
        self, 
        seq_len: int, 
        batch_size: int = 1,
        device: str = "cpu"
    ) -> Int[Tensor, "batch seq"]:
        """Generate samples using ancestral sampling."""
        # Start with absorbing state tokens (or uniform random if not absorbing)
        if self.cfg.transition_type == "absorbing":
            xt = torch.full((batch_size, seq_len), self.absorbing_state_idx, device=device)
        else:
            xt = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len), device=device)
        
        # Ancestral sampling from T to 1
        for t in reversed(range(1, self.cfg.sampling_timesteps + 1)):
            # Denoise
            with torch.no_grad():
                pred_logits = self.denoise(xt, t)
                pred_x0 = torch.argmax(pred_logits, dim=-1)
            
            # For absorbing state, unmask some tokens
            if self.cfg.transition_type == "absorbing" and t > 1:
                # Determine which tokens to keep from prediction and which remain masked
                # This is a simplified version - in practice would use the transition probabilities
                beta_t = self.beta_t(t)
                noise = torch.rand(batch_size, seq_len, device=device)
                keep_mask = (noise > beta_t).float()
                
                # Keep already unmasked tokens, update others according to prediction
                absorbing_mask = (xt == self.absorbing_state_idx).float()
                xt = (xt * (1 - absorbing_mask) + 
                      pred_x0 * absorbing_mask * keep_mask + 
                      xt * absorbing_mask * (1 - keep_mask)).long()
            else:
                xt = pred_x0
        
        return xt


class D3PMConfig(BaseTransformerConfig):
    # Transition matrix types
    transition_type: Literal["uniform", "absorbing", "gaussian", "nearest_neighbor"] = "absorbing"

    # Noise schedule
    noise_schedule: Literal["cosine", "linear", "mutual_info"] = "cosine"

    # Absorbing state (for absorbing transition type)
    absorbing_state_idx: int = -1  # Default to last token in vocab (usually [MASK])

    # For nearest neighbor transition matrices
    nearest_neighbors: int = 5

    # For Gaussian transition matrices
    gaussian_beta_min: float = 1e-4
    gaussian_beta_max: float = 0.02

    # Training parameters
    num_timesteps: int = 1000
    auxiliary_loss_weight: float = 0.01

    # Sampling parameters
    sampling_timesteps: int = 1000

    class Config:
        arbitrary_types_allowed = True


# ---- Adapter + Config for trainer integration ----


class D3PMAdapter(DefaultAdapter):
    Cfg = D3PMConfig

    def build(self, args, cfg: Optional[D3PMConfig]):
        cfg = cfg or D3PMConfig(
            vocab_size=50257,
            n_layers=args.num_layers,
            n_heads=args.num_heads,
            d_model=args.model_dim,
            num_timesteps=1000,
        )
        # Fill in any missing required fields from args
        if not hasattr(cfg, 'n_layers') or cfg.n_layers is None:
            cfg.n_layers = args.num_layers
        if not hasattr(cfg, 'n_heads') or cfg.n_heads is None:
            cfg.n_heads = args.num_heads
        if not hasattr(cfg, 'd_model') or cfg.d_model is None:
            cfg.d_model = args.model_dim
            
        model = D3PM(cfg, is_master_process=True).cuda()
        return model

    def train_step(self, model, inputs, targets, sw_num_blks, *, loss_scale, args):
        loss = model.forward(inputs.view(1, -1), targets.view(1, -1)).loss
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        return model.forward(inputs.view(1, -1), targets.view(1, -1)).loss


