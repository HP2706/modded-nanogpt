import math  
import inspect
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from ops import lm_head_fp8
from utils import next_multiple_of_n
from models.components.shared import ValueEmbedding, Block, MLP, CastedLinear, Rotary, CausalSelfAttention
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
# torch._inductor.config.coordinate_descent_tuning = True # turn this off for a faster compile time (but slightly slower run)

def cosine_normalize(x: Tensor, dim : Optional[int] = None):
    if dim is None:
        dim = -1 #assume its the last dimension
    return x / (x.norm(dim=dim, keepdim=True) + torch.finfo(x.dtype).eps)


class NMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_fc.data /= self.c_fc.data.norm(dim=0, keepdim=True) + torch.finfo(self.c_fc.data.dtype).eps
        
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        
        self.mlp_alpha_init_value = 0.05
        self.mlp_alpha_init_scaling = 1.0 / math.sqrt(dim)
        self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(dim, dtype=torch.float32))


    def forward(self, x):
        x = self.c_fc(x) * self.S_u
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x) * self.S_m * self.scaler
        return x



class NBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, n_layers: Optional[int] = None):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        if layer_idx != 7:
            self.attn = CausalSelfAttention(
                dim=dim, 
                num_heads=num_heads, 
                layer_idx=layer_idx, 
                head_dim=dim//num_heads,
                attn_scale=math.sqrt(dim)
            ) 
        else:
            self.attn = None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        
        # Initialize scaling parameters
        base_scale = 1.0 / math.sqrt(dim)
        self.attn_alpha_init_value = 0.05 if n_layers is None else 1.0 / n_layers
        self.attn_alpha_init_scaling = base_scale
        self.attn_alpha = nn.Parameter(self.attn_alpha_init_scaling * torch.ones(dim, dtype=torch.float32))
        
        self.mlp_alpha_init_value = 0.05 if n_layers is None else 1.0 / n_layers
        self.mlp_alpha_init_scaling = base_scale
        self.mlp_alpha = nn.Parameter(self.mlp_alpha_init_scaling * torch.ones(dim, dtype=torch.float32))

    def forward(self, x, ve, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        
        if self.attn is not None:
            x_a = cosine_normalize(self.attn(x, ve, block_mask))
            lr_attn = torch.abs(self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling))
            x = cosine_normalize(x + lr_attn * (x_a - x))
        
        h_m = cosine_normalize(self.mlp(x))
        lr_mlp = torch.abs(self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling))
        x = cosine_normalize(x + lr_mlp * (h_m - x))
        return x

class NGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        use_fp8: bool = False,
    ):
        super().__init__()
        self.use_fp8 = use_fp8
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx) for layer_idx in range(num_layers)])
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_() # @Grad62304977
        
        self.sz_init_value = 1.00
        self.sz_init_scaling = 1.0 / math.sqrt(model_dim)
        self.sz = torch.nn.Parameter(self.sz_init_scaling*torch.ones(next_multiple_of_n(vocab_size, n=128), dtype=torch.float32))


    def create_block_masks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        any_causal_bm = block_idx[:, None] >= block_idx
        all_causal_bm = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        any_document_bm = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        all_document_bm = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        any_bm = any_causal_bm & any_document_bm
        all_bm = all_causal_bm & all_document_bm
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(any_bm & ~all_bm)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(all_bm)
        def build_bm(sw_num_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(sw_num_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, sw_num_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        long_bm, short_bm = self.create_block_masks(input_seq, sliding_window_num_blocks)

        x = x0 = cosine_normalize(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        
        ve = []
        for val in self.value_embeds(input_seq):
            if val is not None:
                ve.append(cosine_normalize(val))
            else:
                ve.append(None)
        
        
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_masks[i])
        x = cosine_normalize(x)
        
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training and self.use_fp8 else self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        #logits = 30 * torch.sigmoid(logits.float() / 7.5)
        sz = self.sz * (self.sz_init_value/self.sz_init_scaling)
        logits = sz * logits
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss


def log_stats(model : NGPT):    
    _data_dict = {}
    
    for layer_idx in range(0, len(model.blocks)):
        block = model.blocks[layer_idx] 
        sqk = block.sqk * (block.sqk_init_value/block.sqk_init_scaling)
        attn_alpha = block.attn_alpha * (block.attn_alpha_init_value / block.attn_alpha_init_scaling)
        mlp_alpha = block.mlp_alpha * (block.mlp_alpha_init_value / block.mlp_alpha_init_scaling)
        suv = block.suv * (block.suv_init_value/block.suv_init_scaling)
        _data_dict[f"sqk_{layer_idx}"] = torch.mean( sqk )
        _data_dict[f"attn_alpha_{layer_idx}"] = torch.mean( attn_alpha )
        _data_dict[f"mlp_alpha_{layer_idx}"] = torch.mean( mlp_alpha )
        _data_dict[f"suv_{layer_idx}"] = torch.mean( suv )
        
    return _data_dict



def normalize_matrices(model : NGPT):
    model.embed.weight.data.copy_(cosine_normalize(model.embed.weight.data, 1))         # V, n_embd
    model.lm_head.weight.data.copy_(cosine_normalize(model.lm_head.weight.data, 1))           # V, n_embd
    

    for layer_idx in range(0, len(model.blocks)):
        block : NBlock = model.blocks[layer_idx]

        if block.attn is not None:
            #normalize attn weights
            block.attn.qkv_w.data.copy_(cosine_normalize(block.attn.qkv_w.data, 1))             # 3*n_proj, n_embd             # n_proj, n_embd
            block.attn.c_proj.weight.data.copy_(cosine_normalize(block.attn.c_proj.weight.data, 0))   # n_embd, n_proj

        #normalize mlp weights
        block.mlp.c_fc.weight.data.copy_(cosine_normalize(block.mlp.c_fc.weight.data, 1))               # n_proj, n_embd
        block.mlp.c_proj.weight.data.copy_(cosine_normalize(block.mlp.c_proj.weight.data, 0))   # n_embd, n_proj
          

# ---- Adapter + Config for trainer integration ----
from typing import Optional
from pydantic import BaseModel
from trainer_registry import (
    DefaultAdapter,
    _adam_muon_optimizers,
)


class NGPTCfg(BaseModel):
    vocab_size: int = 50257
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    model_dim: Optional[int] = None
    use_fp8: bool = False

    class Config:
        arbitrary_types_allowed = True


class NGPTAdapter(DefaultAdapter):
    Cfg = NGPTCfg

    def build(self, args, cfg: Optional[NGPTCfg]):
        cfg = cfg or NGPTCfg()
        model = NGPT(
            vocab_size=cfg.vocab_size,
            num_layers=cfg.num_layers or args.num_layers,
            num_heads=cfg.num_heads or args.num_heads,
            model_dim=cfg.model_dim or args.model_dim,
            use_fp8=cfg.use_fp8 or args.proj_fp8,
        ).cuda()
        return model

    def create_optimizers(self, model, args, *, rank, world_size, device):
        # Mirror the special-cased NGPT setup
        hidden_matrix_params = [
            p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
        ]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]
        head_params = [model.lm_head.weight]

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
        loss = model.forward(inputs, targets, sw_num_blks)
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        return model.forward(inputs, targets, sw_num_blks)

    def requires_scaled_grad_on_reduce(self) -> bool:
        return False

    def post_optimizer_step(self, model: nn.Module, *, args) -> None:
        normalize_matrices(model)
