import time 
from typing import Optional
from Models.LLMS.LLMBase import TransformerMixin
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from Models.LLMS.LLMBase import BaseTransformerConfig, ModelMixin, ModelOutputMixin
from jaxtyping import Float, Int


class GPTOutput(ModelOutputMixin): 
    pass 

class CausalSelfAttention(nn.Module):
    def __init__(self, config : BaseTransformerConfig):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #type: ignore
        # regularization
        self.num_heads = config.num_heads
        self.d_model = config.d_model

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), num_heads=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention v2
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config : BaseTransformerConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, config.d_model*config.d_mult)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.d_model*config.d_mult, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config : BaseTransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x : Float[Tensor, "B T D"]):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(TransformerMixin):
    def __init__(self, cfg : "GPTConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        # init params
        self.transformer = nn.ModuleDict(dict(
            embedding = self.embedding,
            pos_embed = self.pos_embed,
            h = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layers)]),
            ln_f = nn.LayerNorm(self.cfg.d_model),
        ))
        self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(    
        self, 
        idx : Int[torch.Tensor, "B T D"], 
        targets : Optional[Int[torch.Tensor, "B T D"]] = None
    )->GPTOutput:
        self.check_forward(idx, targets)
        B, T = idx.size()
        assert T <= self.cfg.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.cfg.n_ctx}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.pos_embed(pos) # position embeddings of shape (T, d_model)
        tok_emb = self.transformer.embedding(idx) # token embeddings of shape (B, T, d_model)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return GPTOutput(logits=logits, loss=loss)
        else:
            return GPTOutput(logits=logits)


class GPTConfig(BaseTransformerConfig):
    pass


# ---- Adapter + Config for trainer integration ----
from typing import Optional, Dict
from pydantic import BaseModel
from trainer_registry import (
    ModelAdapter,
    _adam_muon_optimizers,
)
import torch
from torch import nn, Tensor


class GPT2Cfg(BaseModel):
    vocab_size: int = 50257
    n_ctx: Optional[int] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    n_layers: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class GPT2Adapter(ModelAdapter):
    Cfg = GPT2Cfg

    def build(self, args, cfg: Optional[GPT2Cfg]):
        cfg = cfg or GPT2Cfg()
        gcfg = GPTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.n_ctx or args.seq_len,
            d_model=cfg.d_model or args.model_dim,
            num_heads=cfg.num_heads or args.num_heads,
            n_layers=cfg.n_layers or args.num_layers,
            is_training=True,
        )
        model = GPT(gcfg, is_master_process=True).cuda()
        return model

    def create_optimizers(self, model, args, *, rank, world_size, device):
        # Group params: categorize by ndim and names
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
        loss = model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long)).loss
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        return model.forward(inputs.view(1, -1).to(torch.long), targets.view(1, -1).to(torch.long)).loss

    def requires_scaled_grad_on_reduce(self) -> bool:
        return False

    def post_optimizer_step(self, model: nn.Module, *, args) -> None:
        pass
