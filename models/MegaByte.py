from typing import List, Optional
from pydantic import BaseModel, Field
from typing import Literal
from Models.LLMS.configs import ModelConfig
from Models.LLMS.LLMBase import ModelMixin, ModelOutputMixin
from Models.Blocks import GatedMLP, MultiHeadAttention, RMSNorm, RotaryEmbedding
from torch import nn
from torch import Tensor
from jaxtyping import Float, Int
import torch
from einops import rearrange
from utils import text_to_bytes
from torch.nn import functional as F


class MegaByteConfig(ModelConfig):
    patch_size: int = 16
    d_local: int
    d_global_pre_patch: int
    n_layers_d_global: int
    n_layers_d_local: int
    local_n_heads: int
    global_n_heads: int
    d_mult: int  # residual stream = d_mult*d_local/d_global
    is_causal: bool = True
    pad_id: int = 257
    eos_id: int = 258
    vocab_size: int = 256 + 2  # 2 with eos and pad

    @property
    def local_d_head(self) -> int:
        assert self.d_local % self.local_n_heads == 0
        return self.d_local // self.local_n_heads

    @property
    def global_d_head(self) -> int:
        assert self.d_global_pre_patch % self.global_n_heads == 0
        return self.d_global_pre_patch // self.global_n_heads


class MegaByteMultiHeadAttention(nn.Module):
    def __init__(self, cfg: MegaByteConfig, is_global: bool):
        super().__init__()
        self.cfg = cfg
        if is_global:
            dim = cfg.d_global_pre_patch*cfg.patch_size
            n_head = cfg.global_n_heads
            d_head = cfg.global_d_head*cfg.patch_size
        else:
            dim = cfg.d_local
            n_head = cfg.local_n_heads
            d_head = cfg.local_d_head
        
        self.d_model = dim
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(n_head * d_head, dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #type: ignore
        self.n_head = n_head
        self.d_head = d_head

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators 
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_len d_model"],
        attn_mask: Optional[Int[Tensor, "batch sequence_len sequence_len"]] = None,
    ) -> Float[Tensor, "batch sequence_len d_model"]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None if self.cfg.is_causal else attn_mask, 
            is_causal=self.cfg.is_causal
        )
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

    
class MegaByteBlock(nn.Module):
    def __init__(
        self,
        cfg: MegaByteConfig,
        is_global: bool
    ):
        super().__init__()
        self.cfg = cfg
        if is_global:
            dim = self.cfg.d_global_pre_patch*self.cfg.patch_size
        else:
            dim = self.cfg.d_local
        #we do this to ensure multiheadattn uses d_global_pre_patch=d_model
        self.emb = RotaryEmbedding(dim)
        self.attn = MegaByteMultiHeadAttention(cfg, is_global)
        self.mlp = GatedMLP(
            in_features=dim, 
            hidden_features=dim*self.cfg.d_mult, 
            out_features=dim,
            bias=False
        )
        self.is_global = is_global
        self.ln_1 = nn.LayerNorm(dim, eps=cfg.eps)
        self.ln_2 = nn.LayerNorm(dim, eps=cfg.eps)
    
    def forward(
        self,
        x: Float[Tensor, "batch seqlen d_global_pre_patch"]
    ) -> Float[Tensor, "batch seqlen d_global_pre_patch"]:
        #TODO use rotary embeddings
        x = self.ln_1(x)
        x = x + self.attn.forward(x)
        x = self.ln_2(x)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(
        self, 
        cfg: MegaByteConfig,
        is_global: bool
    ):
        super().__init__()
        self.cfg = cfg
        n_layers = cfg.n_layers_d_local if is_global else cfg.n_layers_d_global
        self.local_blocks : List[MegaByteBlock] = nn.ModuleList([ #type: ignore
            MegaByteBlock(cfg, is_global=is_global) for _ in range(n_layers)
        ])

    def forward(
        self,
        x: Float[Tensor, "batch seqlen d_local"]
    ) -> Float[Tensor, "batch seqlen d_local"]:
        for block in self.local_blocks:
            x = block.forward(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, cfg: MegaByteConfig):
        super().__init__()
        self.cfg = cfg
        self.byte_embed = nn.Embedding(cfg.vocab_size, cfg.d_global_pre_patch)
        self.global_pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_global_pre_patch)
        self.global_pad_embed = nn.Parameter(torch.randn(1, 1, cfg.d_global_pre_patch * cfg.patch_size))

    def forward(self, x_bytes: Float[Tensor, "batch seqlen"]) -> Float[Tensor, "batch n_patches patch_size d_global_pre_patch"]:
        assert x_bytes.max() < self.cfg.vocab_size, f"x_bytes.max() {x_bytes.max()} must be less than vocab_size {self.cfg.vocab_size}"
        B, T = x_bytes.shape
        K = T // self.cfg.patch_size
        P = self.cfg.patch_size
        D_G = self.cfg.d_global_pre_patch

        # Pad x_bytes
        g_pad_ids = F.pad(x_bytes.view(B, K, P), (0, 0, 1, 0), value=self.cfg.pad_id)
        
        # Embed and add positional embeddings
        g_pad_embed = self.byte_embed(g_pad_ids.view(B, -1))
        positions = torch.arange(g_pad_embed.size(1), device=g_pad_embed.device).unsqueeze(0)
        positions = positions % self.cfg.n_ctx  
        g_pad_embed += self.global_pos_embed(positions)
        
        # Reshape to patches
        global_in = g_pad_embed.view(B, K + 1, P * D_G)
        x = global_in[:, :-1]
        return x

from einops.layers.torch import Rearrange

class MegaByte(ModelMixin):
    def __init__(
        self, 
        cfg: MegaByteConfig, 
        is_master_process: bool = True
    ):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.global_model = Transformer(cfg, is_global=True)
        self.local_model = Transformer(cfg, is_global=False)
        self.to_local_proj = nn.Linear(cfg.d_global_pre_patch, cfg.d_local)
        self.l_embedder = nn.Embedding(cfg.vocab_size, cfg.d_local)
        self.local_pad = nn.Parameter(torch.randn(1, 1, cfg.d_local))
        self.lm_head = nn.Linear(cfg.d_local, cfg.vocab_size)
    
    def _init_weights(self):
        pass

    def forward(
        self,
        ids: Float[Tensor, "batch seqlen"],
        target_tokens: Optional[Float[Tensor, "batch seqlen"]] = None
    ) -> ModelOutputMixin:
        assert ids.shape[1] % self.cfg.patch_size == 0, "ids.shape[1] must be divisible by patch_size"
        if target_tokens is not None:
            assert target_tokens.shape[1] % self.cfg.patch_size == 0, "target_tokens.shape[1] must be divisible by patch_size"
        T = ids.size(1)
        P = self.cfg.patch_size
        K = T//P
        B = ids.size(0)
        x = self.patch_embed(ids)
        global_out = self.global_model.forward(x)
        local_in = global_out.view(B, -1, self.cfg.patch_size, self.cfg.d_global_pre_patch)
        x = self.to_local_proj.forward(local_in) 

        #prepare local pos embeddings
        l_input_ids = rearrange(ids, "B (K P) -> (B K) P", B=B, K=K, P=P)
        l_input_ids = F.pad(l_input_ids, (1, -1), value=self.cfg.pad_id)
        l_embed = self.l_embedder(l_input_ids).view(B, K, P, self.cfg.d_local)

        x = x + l_embed

        #we process all K patches in parallel by rolling into batch_dimension
        logits = self.local_model.forward(x.view(B*K, P, self.cfg.d_local))
        if target_tokens is not None:
            logits = self.lm_head(logits)
            logits = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits, target_tokens.view(-1))
            return ModelOutputMixin(
                loss=loss,
                logits=x
            )
        return ModelOutputMixin(logits=x)
