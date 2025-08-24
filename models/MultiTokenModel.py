import torch
import torch.nn as nn
import math
from typing import List, Optional, cast
import torch.nn.functional as F
from Models.Blocks import GatedMLP, RMSNorm, RotaryEmbedding
from Models.LLMS.LLMBase import ModelMixin
from torch import Tensor
from jaxtyping import Float, Int
from Models.LLMS.configs import BaseTransformerConfig
from Models.LLMS.gpt2 import Block, GPT, GPTOutput
from jaxtyping import jaxtyped, Float, Int


class MultiTokenModelOutput(GPTOutput):
    mtp_loss : Optional[Float[Tensor, ""]] = None

class MultiTokenModelConfig(BaseTransformerConfig):
    n_mtp_modules : int
    mtp_lambda : float


#deepseek v3 Multi token prediction model
class MTPModule(nn.Module):
    def __init__(self, cfg : MultiTokenModelConfig):
        super().__init__()
        self.cfg = cfg
        self.block = Block(cfg)
        self.ln_1 = RMSNorm(cfg.d_model)
        self.ln_2 = RMSNorm(cfg.d_model)
        self.lin_proj = nn.Linear(cfg.d_model*2, cfg.d_model)
        
    def forward(
        self, 
        processed_tokens : Float[Tensor, "batch 1 d_model"],
        new_token_embeddings : Float[Tensor, "batch 1 d_model"]
        ) -> Float[Tensor, "batch 1 d_model"]:
        h_prev = self.ln_1(processed_tokens)
        emb_new = self.ln_2(new_token_embeddings)
        combined = torch.cat([h_prev, emb_new], dim=-1)
        h_prime = self.lin_proj(combined)
        h_k = self.block(h_prime)
        return h_k
        
class MultiTokenModel(GPT):
    cfg : MultiTokenModelConfig
    mtp_module : MTPModule
    def __init__(self, cfg : MultiTokenModelConfig, is_master_process : bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        self.mtp_modules = nn.ModuleList([MTPModule(cfg) for _ in range(cfg.n_mtp_modules)])
    
    def forward(
        self, 
        idx : Int[torch.Tensor, "B T"], 
        targets : Optional[Int[torch.Tensor, "B T"]] = None,    
    ) -> MultiTokenModelOutput:
        self.check_forward(idx, targets)
        
        B, T = idx.size()
        assert T <= self.cfg.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.cfg.n_ctx}"
        # forward the token and posisition embeddings
        
        T_model = T - self.cfg.n_mtp_modules
        
        
        pos = torch.arange(0, T_model, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.pos_embed(pos) # position embeddings of shape (T, d_model)
        tok_emb = self.transformer.embedding(idx) # token embeddings of shape (B, T, d_model)
        
        tok_mtp = tok_emb[:, T_model:, :] # (B, T - T_model, d_model)
        x = tok_emb[:, :T_model, :] + pos_emb # (B, T_model, d_model)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        mtp_losses = []
        h = x[:, T_model-1:T_model, :] # get the last token
        for k, mtp_module in enumerate(self.mtp_modules, start=1):
            # Get previous representations for positions 1 to T-k

            mtp_module = cast(MTPModule, mtp_module)
            # Get embeddings for the k-th next tokens
            next_token_emb = tok_mtp[:, k-1:k, :]
            
            # Forward through MTP module (Equations 21-22)
            h = mtp_module.forward(h, next_token_emb)
            
            # Get predictions for k-th next tokens (Equation 23)
            mtp_logits = self.lm_head(h)  # Shared output head
            
            if targets is not None:
                # Equation 24: MTP loss for depth k
                mtp_targets = targets[:, T_model+k-1:T_model+k]
                mtp_loss = F.cross_entropy(
                    mtp_logits.view(-1, mtp_logits.size(-1)),
                    mtp_targets.view(-1)
                )
                mtp_losses.append(mtp_loss)
        
        if targets is not None:
            # Main language modeling loss
            targets_lm = targets[:, :T_model]
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_lm.view(-1))
            
            # Average MTP losses and combine with main loss
            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()
                loss = lm_loss + self.cfg.mtp_lambda * mtp_loss  # λ from paper
            else:
                loss = lm_loss
                
            return MultiTokenModelOutput(logits=logits, loss=loss, mtp_loss=mtp_loss)
        else:
            return MultiTokenModelOutput(logits=logits)
        
        
        