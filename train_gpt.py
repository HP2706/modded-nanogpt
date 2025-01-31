import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
from typing import Optional, Union
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
torch.empty(1, device=device, requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # turn this off for a faster compile time (but slightly slower run)

# -----------------------------------------------------------------------------
# Custom operators : FP8 matmul for lm_head by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.mul(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.mul(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.t(),
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(1 / x_s, dtype=torch.float32),
            scale_b=x.new_tensor(1 / w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.t(), x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(1 / x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(1 / w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(1 / grad_s, dtype=torch.float32)
        grad_f8 = grad.mul(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.t().contiguous().t(),
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.t().contiguous(),
            grad_f8.t().contiguous().t(),
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).t()
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

def lm_head_fp8(x: Tensor, w: Tensor) -> Tensor:
    _x = x.flatten(0, -2)
    out: Tensor = torch.ops.nanogpt.mm(_x, w, x_s=2.0, w_s=32.0, grad_s=2.0**29)[0]
    return out.reshape(*x.shape[:-1], -1)

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device=device)
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x):
    print0(f"Starting norm operation with tensor shape: {x.shape}")
    result = F.rms_norm(x, (x.size(-1),))
    print0("Completed norm operation")
    return result

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, head_dim=128):
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
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, layer_idx) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        print0(f"Block forward: input shape {x.shape}")
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            print0("Starting attention")
            x = x + self.attn(norm(x), ve, block_mask)
            print0("Completed attention")
        print0("Starting MLP")
        x = x + self.mlp(norm(x))
        print0("Completed MLP")
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for _ in range(3)])

    def forward(self, input_seq) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        return ve

# -----------------------------------------------------------------------------
# The main model

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

class MTP(torch.nn.Module):
    def __init__(
        self, 
        n_tokens : int,
        d_hidden : int,
        d_vocab : int,
        use_liger : bool = True,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_hidden = d_hidden
        self.d_vocab = d_vocab
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(d_hidden, d_vocab) for _ in range(n_tokens)
        ])
        self.use_liger = use_liger
        if use_liger:
            self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        

    def forward(self, hidden_states : torch.Tensor, labels : torch.LongTensor):
        '''
        hidden_states : (B, T, H)
        labels : (B, T+N_tokens)
        
        
        this function computes the gradient of the loss 
        '''
        assert hidden_states.shape[1] == labels.shape[1] - self.n_tokens
        
        shared_trunk = hidden_states.detach()
        shared_trunk.requires_grad = True
        
        loss_accum = 0
        for i in range(self.n_tokens):
            # remove the first i tokens
            shift_labels = labels[..., (1 + i) :].contiguous().view(-1) 
            head : torch.nn.Linear = self.heads[i]
            #TODO can this be done in fp8?
            #TODO 30 * torch.sigmoid(logits.float() / 7.5) tanh softcapping??
            if self.use_liger:
                loss = self.loss_fn.forward(
                    head.weight,
                    shared_trunk, # merge batch and sequence dimensions
                    shift_labels,
                    bias=head.bias,
                )
                
            else:
                #TODO use fp8 matmul??
                logits = head.forward(shared_trunk)
                loss =self.loss_fn.forward(logits.view(-1, self.d_vocab), shift_labels)
            
            #NOTE we call backward multiple times to accumulate the gradients to d(the shared trunk)
            loss.backward()
            loss_accum += loss.detach() # We only need this for logging
            
            
        #hidden_states.backward(gradient=shared_trunk.grad)
        return loss_accum, shared_trunk.grad
        

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        num_layers: int, 
        num_heads: int, 
        model_dim: int, 
        n_mtp_tokens: int, 
        use_liger: bool = True,
        use_mtp: bool = True
    ):
        super().__init__()
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
        
        self.use_mtp = use_mtp
        if use_mtp:
            self.mtp = MTP(
                n_tokens=n_mtp_tokens,
                d_hidden=model_dim,
                d_vocab=next_multiple_of_n(vocab_size, n=128),
                use_liger=use_liger,
            )
        else:
            self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
            self.lm_head.weight.detach().zero_() # @Grad62304977
            
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
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
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

    def forward(
        self, 
        input_seq: Tensor, 
        target_seq: Tensor, 
        sliding_window_num_blocks: Tensor
    )-> Tensor | tuple[Tensor, Tensor]:
        assert input_seq.ndim == 1

        long_bm, short_bm = self.create_block_masks(input_seq, sliding_window_num_blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        ve = self.value_embeds(input_seq)
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
        x = norm(x)
        
        if not self.use_mtp:
            logits = lm_head_fp8(x, self.lm_head.weight) if self.training else self.lm_head(x)
            # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
            logits = 30 * torch.sigmoid(logits.float() / 7.5)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
            return loss
        else:
            # we backpropagate the loss by doing x.backward(gradient=shared_trunk_grad)
            # instead of loss.backward()
            # we keep the loss for logging
            loss, shared_trunk_grad = self.mtp.forward(x, target_seq)
            return loss, x, shared_trunk_grad 
            

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    print0(f"Starting data generator with batch_size={batch_size}, rank={rank}, world_size={world_size}")
    files = sorted(Path.cwd().glob(filename_pattern))
    print0(f"Found {len(files)} files matching pattern: {filename_pattern}")
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    print0("Loading first data shard...")
    tokens, pos = _load_data_shard(next(file_iter)), 0
    print0(f"Loaded initial shard with {len(tokens)} tokens")
    while True:
        if pos + batch_size + 1 >= len(tokens):
            print0("Loading next data shard...")
            tokens, pos = _load_data_shard(next(file_iter)), 0
            print0(f"Loaded new shard with {len(tokens)} tokens")
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main
# if we use python instead of torchrun, we need to set the rank and world size manually
if os.environ.get("RANK") is None:
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device(device, int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()

master_process = (rank == 0) # this process will do logging, checkpointing etc.

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization

    seq_len = 64*1024 # FlexAttention sequence length
    batch_size = world_size*seq_len   #TODO back to: 64*1024 # batch size in tokens
    num_iterations = 1393 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    # implementation
    save_checkpoint = False
    use_liger = False
    use_adam_mini = False 
    use_mtp = False
    num_layers = 12
    num_heads = 6
    model_dim = 768
    n_mtp_tokens : Optional[int] = 12
    use_wandb = True
    torch_compile = False # for faster iteration speed
args = Hyperparameters()

print("args", args)

def gen_name(args: Hyperparameters, uuid: uuid.UUID):
    return (
        f'{uuid}_torch_compile={args.torch_compile}_batch_size={args.batch_size}'
        f'_use_liger={args.use_liger}'
        f'_use_mtp={args.use_mtp}'
        f'_use_adam_mini={args.use_adam_mini}'
        f'_num_layers={args.num_layers}'
        f'_num_heads={args.num_heads}'
        f'_model_dim={args.model_dim}'
        f'_n_mtp_tokens={args.n_mtp_tokens}'
    )

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
    if args.use_wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = 'a3469eb2df23f67e4d6907ebacf50ffb4ee664f7'
        name = gen_name(args, run_id)
        wandb.init(
            project="modded-nanogpt", 
            name=name,
            config=args
        )
def print0(_out_dict : Union[dict, str], console=False):
    if master_process:
        if isinstance(_out_dict, dict):
            if args.use_wandb:
                wandb.log(_out_dict)
            
            s = ""
            for k, v in _out_dict.items():
                s += f"{k}: {v} "
        else:
            s = _out_dict
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

print0("Loading data", console=True)
# load data
train_loader = distributed_data_generator(args.train_files, args.batch_size, rank, world_size)

print0("Creating model", console=True)
model = GPT(
    vocab_size=50257, 
    num_layers=args.num_layers, 
    num_heads=args.num_heads,
    model_dim=args.model_dim, 
    n_mtp_tokens=args.n_mtp_tokens, 
    use_liger=args.use_liger, 
    use_mtp=args.use_mtp
).cuda()

print0("Model created", console=True)
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
if not args.use_adam_mini:

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    optimizers = [optimizer1, optimizer2]
else:
    from adam_mini import Adam_mini
    embed_params = [(n,p) for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [(n,p) for n, p in model.named_parameters() if p.ndim < 2]
    head_params = [('lm_head.weight', model.lm_head.weight)]
    optimizers = [Adam_mini(
        dict(embed_params, lr=0.008),
        dict(scalar_params, lr=0.04),
        dict(head_params, lr=0.008),
        weight_decay=0.01
    )]

# learning rate schedule: stable then decay
def get_lr(step: int):
    t = 1 - step / args.num_iterations # time remaining in training
    assert 1 >= t >= 0
    w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
    return w * 1.0 + (1 - w) * 0.1
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
@lru_cache(1)
def sw_num_blks(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

if args.torch_compile:
    print0("Torch compile enabled, compiling model...", console=True)
    time_start = time.perf_counter()
    model: GPT = torch.compile(model)
    time_end = time.perf_counter()
    print0(f"Torch compile time: {time_end - time_start:.2f} seconds", console=True)
else:
    print0("Torch compile disabled", console=True)
    
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
print0(f"Starting training loop for {train_steps} steps", console=True)
for step in range(train_steps + 1):
    print0(f"Beginning step {step}", console=True)
    last_step = (step == train_steps)
    
    if step == 10:
        print0("Resetting timing measurements", console=True)
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float("nan") if step <= 11 else (step - 10) + 1

    window_size = next_multiple_of_n(1728 * step / train_steps, n=128)
    print0(f"Window size for step {step}: {window_size}", console=True)

    # Validation section
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        print0(f"Starting validation at step {step}", console=True)
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.seq_len
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            from tqdm import tqdm
            for val_step in tqdm(range(val_steps), desc="Validation steps", leave=False, disable=not master_process):
                x, y = next(val_loader)
                val_loss += model.forward(x, y, sw_num_blks(window_size))
        
        print0("Validation complete, reducing loss", console=True)
        val_loss /= val_steps
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0({"step": step, "val_loss": val_loss, "train_time": training_time_ms}, console=True)
        
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        print0("Reached last step, breaking", console=True)
        break

    # Training section
    print0(f"Loading training batch for step {step}", console=True)
    inputs, targets = next(train_loader)
    print0(f"Processing {len(inputs.split(args.seq_len))} sequences", console=True)
    print0(f"Input shape: {inputs.shape}, Target shape: {targets.shape}", console=True)
    
    for i, (input_seq, target_seq) in enumerate(zip(inputs.split(args.seq_len), targets.split(args.seq_len))):
        print0(f"Processing sequence {i+1}/{len(inputs.split(args.seq_len))}", console=True)
        print0(f"Sequence shapes - input: {input_seq.shape}, target: {target_seq.shape}", console=True)
        out = model.forward(input_seq, target_seq, sw_num_blks(window_size))
        print0(f"Forward pass completed for sequence {i+1}", console=True)
        if isinstance(out, tuple):
            loss, shared_trunk_grad = out
            print0(f"Starting backward pass for sequence {i+1}", console=True)
            shared_trunk_grad.backward()
            print0(f"Completed backward pass for sequence {i+1}", console=True)
        else:
            loss = out
            
    print0("Reducing gradients", console=True)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
    print0("Updating optimizers", console=True)
    frac = min(step / 300, 1)
    for group in optimizer2.param_groups:
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
        
    model.zero_grad(set_to_none=True)
    
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"Completed step {step}", console=True)

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
    console=True
)
dist.destroy_process_group()
