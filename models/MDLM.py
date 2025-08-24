import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from jaxtyping import Float, Int
from Models.LLMS.LLMBase import TransformerMixin, ModelOutputMixin
from Models.LLMS.configs import BaseTransformerConfig
from Models.LLMS.gpt2 import Block
import math

class MDLMOutput(ModelOutputMixin):
    logits: Tensor
    loss: Optional[Tensor] = None
    diffusion_loss: Optional[Tensor] = None

class MDLM(TransformerMixin):
    def __init__(self, cfg: "MDLMConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        
        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            embedding=self.embedding,
            pos_embed=self.pos_embed,
            h=nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)]),
            ln_f=nn.LayerNorm(self.cfg.d_model),
        ))
        
        # Output head for denoising
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        
        # Weight sharing
        self.transformer.embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
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
    
    def alpha_t(self, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch"]:
        """Compute alpha_t based on the noise schedule."""
        if self.cfg.alpha_type == "cosine":
            return torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        elif self.cfg.alpha_type == "linear":
            return 1.0 - t
        else:
            raise ValueError(f"Unknown alpha_type: {self.cfg.alpha_type}")
    
    def alpha_t_derivative(self, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch"]:
        """Compute derivative of alpha_t."""
        if self.cfg.alpha_type == "cosine":
            return -math.pi / 2 * torch.sin((t + 0.008) / 1.008 * math.pi / 2) * \
                   torch.cos((t + 0.008) / 1.008 * math.pi / 2) * (1.0 / 1.008)
        elif self.cfg.alpha_type == "linear":
            return -torch.ones_like(t)
        else:
            raise ValueError(f"Unknown alpha_type: {self.cfg.alpha_type}")
    
    def add_noise(
        self, 
        x: Int[Tensor, "batch seq"], 
        t: Float[Tensor, "batch"]
    ) -> Tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
        """Add noise to input tokens according to the forward process."""
        batch_size, seq_len = x.shape
        
        # Expand t to match x dimensions
        t_expanded = t.unsqueeze(-1).expand(-1, seq_len)
        
        # Compute alpha_t
        alpha_t = self.alpha_t(t)
        alpha_t_expanded = alpha_t.unsqueeze(-1).expand(-1, seq_len)
        
        # Sample random noise
        noise = torch.rand(batch_size, seq_len, device=x.device)
        
        # Create mask tokens (assuming last token in vocab is mask token)
        mask_token = self.cfg.vocab_size - 1
        mask = (noise > alpha_t_expanded).long()
        
        # Apply masking
        x_noisy = x * (1 - mask) + mask_token * mask
        
        return x_noisy, mask
    
    def subs_parameterization(
        self, 
        logits: Float[Tensor, "batch seq vocab"], 
        zt: Int[Tensor, "batch seq"],
        mask_token: int
    ) -> Float[Tensor, "batch seq vocab"]:
        """Apply SUBS parameterization to ensure proper masking behavior."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Zero masking probabilities: set logit for mask token to -inf
        logits_mask = (zt == mask_token).unsqueeze(-1).expand(-1, -1, vocab_size)
        logits = torch.where(logits_mask, logits, logits)
        logits[..., mask_token] = torch.where(
            logits_mask[..., mask_token], 
            torch.full_like(logits[..., mask_token], float('-inf')), 
            logits[..., mask_token]
        )
        
        # Carry-over unmasking: copy unmasked tokens
        unmasked_mask = (zt != mask_token).unsqueeze(-1).expand(-1, -1, vocab_size)
        if unmasked_mask.any():
            # Create one-hot encoding of unmasked tokens
            zt_one_hot = F.one_hot(zt, vocab_size).float()
            logits = torch.where(unmasked_mask, zt_one_hot, logits)
        
        return logits
    
    def denoise(
        self, 
        zt: Int[Tensor, "batch seq"], 
        t: Float[Tensor, "batch"]
    ) -> Float[Tensor, "batch seq vocab"]:
        """Denoise noisy tokens at time t."""
        B, T = zt.size()
        
        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=zt.device)
        pos_emb = self.transformer.pos_embed(pos)
        tok_emb = self.transformer.embedding(zt)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Output logits
        logits = self.lm_head(x)
        
        # Apply SUBS parameterization if enabled
        if self.cfg.use_subs_parameterization:
            mask_token = self.cfg.vocab_size - 1
            logits = self.subs_parameterization(logits, zt, mask_token)
        
        return logits
    
    def compute_diffusion_loss(
        self, 
        x: Int[Tensor, "batch seq"], 
        t: Optional[Float[Tensor, "batch"]] = None
    ) -> Float[Tensor, ""]:
        """Compute the continuous-time diffusion loss."""
        batch_size, seq_len = x.shape
        
        # Sample random time steps if not provided
        if t is None:
            t = torch.rand(batch_size, device=x.device)
        
        # Add noise to input
        x_noisy, mask = self.add_noise(x, t)
        
        # Denoise
        logits = self.denoise(x_noisy, t)
        
        # Compute loss for masked tokens only
        # The loss is the weighted cross-entropy between logits and true tokens
        alpha_t = self.alpha_t(t)
        alpha_t_derivative = self.alpha_t_derivative(t)
        
        # Weight for the loss (from continuous-time NELBO)
        weight = -alpha_t_derivative / (1 - alpha_t)
        
        # Expand weight to match sequence dimensions
        weight_expanded = weight.unsqueeze(-1).expand(-1, seq_len)
        
        # Compute cross-entropy loss for masked positions only
        ce_loss = F.cross_entropy(
            logits.view(-1, self.cfg.vocab_size), 
            x.view(-1), 
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Apply mask to compute loss only for masked tokens
        masked_loss = ce_loss * mask
        
        # Weighted loss
        weighted_loss = weight_expanded * masked_loss
        
        # Average over batch and sequence (only masked tokens)
        mask_sum = mask.sum()
        if mask_sum > 0:
            diffusion_loss = weighted_loss.sum() / mask_sum
        else:
            diffusion_loss = torch.tensor(0.0, device=x.device)
        
        return diffusion_loss
    
    def forward(
        self, 
        idx: Int[Tensor, "batch seq"], 
        targets: Optional[Int[Tensor, "batch seq"]] = None,
        compute_diffusion_loss: bool = True
    ) -> MDLMOutput:
        """Forward pass for MDLM."""
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
        
        # Compute diffusion loss
        diffusion_loss = None
        if compute_diffusion_loss:
            diffusion_loss = self.compute_diffusion_loss(idx)
        
        # Total loss
        loss = None
        if lm_loss is not None and diffusion_loss is not None:
            loss = lm_loss + diffusion_loss
        elif lm_loss is not None:
            loss = lm_loss
        elif diffusion_loss is not None:
            loss = diffusion_loss
        
        return MDLMOutput(
            logits=logits, 
            loss=loss, 
            diffusion_loss=diffusion_loss
        )
    
    def sample(
        self, 
        seq_len: int, 
        batch_size: int = 1,
        device: str = "cpu"
    ) -> Int[Tensor, "batch seq"]:
        """Generate samples using ancestral sampling."""
        # Start with all mask tokens
        mask_token = self.cfg.vocab_size - 1
        x_t = torch.full((batch_size, seq_len), mask_token, device=device)
        
        # Sample time steps
        if self.cfg.continuous_time:
            # Continuous time sampling
            time_steps = torch.linspace(1.0, 0.0, self.cfg.num_sampling_steps + 1, device=device)
        else:
            # Discrete time sampling
            time_steps = torch.linspace(1.0, 0.0, self.cfg.num_sampling_steps + 1, device=device)
        
        # Ancestral sampling
        for i in range(len(time_steps) - 1):
            t = time_steps[i]
            t_next = time_steps[i + 1]
            
            # Denoise
            with torch.no_grad():
                logits = self.denoise(x_t, t.expand(batch_size))
            
            # Sample from logits
            probs = F.softmax(logits, dim=-1)
            x_t_new = torch.multinomial(probs.view(-1, self.cfg.vocab_size), 1).view(batch_size, seq_len)
            
            # Apply SUBS parameterization constraints
            if self.cfg.use_subs_parameterization:
                # Copy unmasked tokens
                unmasked = (x_t != mask_token)
                x_t_new = torch.where(unmasked, x_t, x_t_new)
            
            x_t = x_t_new
        
        return x_t
    
    def semi_autoregressive_sample(
        self, 
        prefix: Optional[Int[Tensor, "batch prefix_len"]] = None,
        total_len: int = 1024,
        batch_size: int = 1,
        device: str = "cpu"
    ) -> Int[Tensor, "batch total_len"]:
        """Generate samples using semi-autoregressive decoding."""
        if prefix is None:
            # Initialize with mask tokens
            mask_token = self.cfg.vocab_size - 1
            prefix = torch.full((batch_size, self.cfg.sar_chunk_size), mask_token, device=device)
        
        batch_size, prefix_len = prefix.shape
        result = prefix
        
        # Generate in chunks
        while result.shape[1] < total_len:
            # Generate next chunk
            chunk_len = min(self.cfg.sar_chunk_size, total_len - result.shape[1])
            chunk = self.sample(
                seq_len=chunk_len, 
                batch_size=batch_size, 
                device=device
            )
            
            # Concatenate with existing result
            result = torch.cat([result, chunk], dim=1)
        
        return result[:, :total_len]


class MDLMConfig(BaseTransformerConfig):
    # Noise schedule parameters
    alpha_type: str = "cosine"  # cosine or linear
    alpha_min: float = 0.001    # minimum alpha value
    alpha_max: float = 1.0      # maximum alpha value

    # Diffusion parameters
    continuous_time: bool = True  # Use continuous time formulation
    use_subs_parameterization: bool = True  # Use SUBS parameterization

    # Sampling parameters
    num_sampling_steps: int = 100  # Number of steps for ancestral sampling
    use_low_discrepancy_sampler: bool = True  # Use low discrepancy sampler

    # Semi-autoregressive decoding
    sar_chunk_size: int = 512  # Chunk size for semi-autoregressive decoding

    class Config:
        arbitrary_types_allowed = True
