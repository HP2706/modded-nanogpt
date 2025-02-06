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
from models.shared import ValueEmbedding, Block, MLP, CastedLinear, Rotary
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



class CausalSelfAttention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        layer_idx: int, 
        head_dim=128,
        use_liger=False,
        attn_scale=0.12,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = cosine_normalize(q), cosine_normalize(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

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
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
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
          
#https://github.com/NVIDIA/ngpt/blob/main/model.py
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = False#fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer