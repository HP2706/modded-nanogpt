from typing import cast
import einops
from einops import rearrange, reduce, einsum, repeat, pack, unpack
from torch import arange
import torch
import einx
from jaxtyping import Float
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch import nn, Tensor
from utils import next_multiple_of_n, round_down_multiple, round_up_multiple
from native_sparse_attention_pytorch import SparseAttention
from native_sparse_attention_pytorch.native_sparse_attention import create_fine_mask, interpolate_1d, max_neg_value, pad_at_dim, straight_through, attend
from ops import lm_head_fp8
from models.shared import ValueEmbedding, norm, CastedLinear, MLP, create_block_masks, Rotary
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask
from typing import Callable

MASK_FN = Callable[[Tensor, int], BlockMask]

def create_fine_mask(seq_len, fine_block_size) -> MASK_FN:

    def inner(selected_block_indices: Tensor, num_grouped_queries = 1):
        device = selected_block_indices.device
        batch, kv_heads = selected_block_indices.shape[:2]

        one_hot_selected_block_indices = torch.zeros((*selected_block_indices.shape[:-1], seq_len // fine_block_size), device = device, dtype = torch.bool)
        one_hot_selected_block_indices.scatter_(-1, selected_block_indices, True)

        def fine_mask(b_idx, h_idx, q_idx, kv_idx):

            compressed_q_idx = q_idx // fine_block_size
            compressed_kv_idx = kv_idx // fine_block_size
            kv_head_idx = h_idx // num_grouped_queries

            is_selected = one_hot_selected_block_indices[b_idx, kv_head_idx, q_idx, compressed_kv_idx]

            causal_mask = q_idx >= kv_idx
            block_diagonal = compressed_q_idx == compressed_kv_idx

            return (causal_mask & (block_diagonal | is_selected))

        block_mask = create_block_mask(fine_mask, B = batch, H = kv_heads * num_grouped_queries, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
        return block_mask

    return inner



class CompressMLP(nn.Module):
    def __init__(self, head_dim: int, compress_block_size: int, expansion_factor : int = 4):
        super().__init__()
        
        compress_dim = compress_block_size * head_dim
        hidden_dim = compress_dim * expansion_factor
        self.c_fc = CastedLinear(compress_dim, hidden_dim)
        self.c_proj = CastedLinear(hidden_dim, head_dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class NSA_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        layer_idx: int,
        sliding_window_size: int = 32,
        compress_block_size: int = 4,
        selection_block_size: int = 4,
        num_selected_blocks: int = 4,
        num_compressed_mem_kv: int = 1,
        query_heads_share_selected_kv : bool = True,
        attn_scale: float = 0.12,
        use_triton_kernel : bool = False,
        use_fine_flex_attention : bool = False,
        use_diff_topk : bool = True,
    ):
        '''
        NSA sparse attention as by https://arxiv.org/abs/2502.11089
        implementation inspired by https://github.com/lucidrains/native-sparse-attention-pytorch.git
        
        Idea:
        - we implement a combination of different attention variants
        - fine selection attention new
        - compress attention new
        - sliding window attention(not new)

        compressed attention: 
            we do a convolution over our tokens to get a smaller number 
            of tokens we then perform normal attention with. 
            
        fine selection attention:
            we select the tokens with the topk scores from the compressed attention
            and perform standard softmax attention on this block of tokens.
        '''
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim_head
        hdim = self.num_heads * self.head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim_head)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977
        # scale the attention logits by given constant, instead of the default head_dim**-0.5, by @leloykun
        # inspired by learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = attn_scale
        self.num_selected_blocks = num_selected_blocks

        # NEW STUFF
        
        assert not (use_fine_flex_attention is True and use_triton_kernel is True), "Cannot use both fine flex attention and triton kernel"
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.num_grouped_queries = heads 

        self.use_fine_flex_attention = use_fine_flex_attention
        self.use_triton_kernel = use_triton_kernel
        # used for compression of both k and v cache
        self.k_compress = CompressMLP(dim_head, compress_block_size=compress_block_size)
        self.v_compress = CompressMLP(dim_head, compress_block_size=compress_block_size)
        self.compress_block_size = compress_block_size
        self.selection_block_size = selection_block_size
        
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, heads, num_compressed_mem_kv, dim_head))
        self.k_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))
        self.flip_h_seq_dim = Rearrange('b n h d -> b h n d')

        strategy_combine_mlp = nn.Linear(dim, 3 * heads)

        # init to sliding windows first, as network tends to pick up on local patterns first before distant ones

        nn.init.zeros_(strategy_combine_mlp.weight)
        strategy_combine_mlp.bias.data.copy_(torch.tensor([-2., -2., 2.] * heads))

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n = compress_block_size)

        self.use_diff_topk = use_diff_topk
        # combining heads
        self.combine_heads = nn.Linear(hdim, dim, bias = False)

    def compressed_attention(
        self,
        q: Float[Tensor, 'b h t d_head'],
        k: Float[Tensor, 'b h t d_head'],
        v: Float[Tensor, 'b h t d_head'],
        mem_ck : Float[Tensor, 'b h n_d'],
        mem_cv : Float[Tensor, 'b h n_d'],
        num_mem_compress_kv : int,
        num_compress_blocks : int,
        compress_divisible_seq_len : int,
        
        device: torch.device
    ) -> tuple[Float[Tensor, 'b h t d'], Float[Tensor, 'b h t t']]:
        T = q.size(2) # sequence length
        
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)


        k  = k[..., :compress_divisible_seq_len, :, :] + k_pos.unsqueeze(0) # expand to batch dim
        v  = v[..., :compress_divisible_seq_len, :, :] + v_pos.unsqueeze(0)
        
        k_compress_input = self.split_compress_window(k)
        v_compress_input = self.split_compress_window(v)

        k_compress_input = rearrange(k_compress_input, 'b h w n d -> b h w (n d)') # we concatenate tokens
        v_compress_input = rearrange(v_compress_input, 'b h w n d -> b h w (n d)')
        
        ck = self.k_compress(k_compress_input)   # Equation (7) of the Native Sparse Attention paper
        cv = self.v_compress(v_compress_input)
        cq = q

        ck = torch.cat((mem_ck, ck), dim = -2)
        cv = torch.cat((mem_cv, cv), dim = -2)

        cq_seq = torch.arange(T, device = device)
        ck_seq = ((torch.arange(num_compress_blocks, device = device) + 1) * self.compress_block_size) - 1
        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value = -1)

        cmask = einx.less('j, i -> i j', ck_seq, cq_seq)

        compressed_attn_out, csim = attend(cq, ck, cv, mask = cmask, return_sim = True)
        return compressed_attn_out, csim
    
    def fine_attention(
        self,
        fq: Float[Tensor, 'b h t d_head'],
        fk: Float[Tensor, 'b h t d_head'],
        fv: Float[Tensor, 'b h t d_head'],
        csim : Float[Tensor, 'b h t t'],
        num_mem_compress_kv : int,
        num_compress_blocks : int,
        num_fine_blocks : int,
        fine_divisible_seq_len : int,
        device: torch.device,
        disable_triton_kernel : bool = False,
    ) -> Float[Tensor, 'b h t d_head']:
        B, S = fq.size(0), fq.size(2)

        importance_scores = csim[..., num_mem_compress_kv:]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0


        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)
            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1


        # handle if compress block size does not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length, then average within each fine / selection block size - pad on the right to 0s, which should be fine as sliding window convers the local anyways

        if has_selected_kv_for_fine_attn:
            if self.compress_block_size != self.selection_block_size:

                compress_seq_len = num_compress_blocks * self.compress_block_size
                if self.interpolated_importance_score:
                    importance_scores = interpolate_1d(importance_scores, compress_seq_len)
                else:
                    importance_scores = repeat(importance_scores, '... j -> ... (j block_size)', block_size = self.compress_block_size)
                padding = fine_divisible_seq_len - compress_seq_len

                fine_query_seq_len = importance_scores.shape[-2]

                importance_scores = F.pad(importance_scores, (0, padding))

                # mask out the diagonal since block causal is included by default for fine attending

                block_causal_mask = torch.ones((num_fine_blocks,) * 2, device = device, dtype = torch.bool).tril(-1)
                block_causal_mask = repeat(block_causal_mask, 'i j -> (i n1) (j n2)', n1 = self.selection_block_size, n2 = self.selection_block_size)
                block_causal_mask = block_causal_mask[:fine_query_seq_len]

                importance_scores = importance_scores.masked_fill(~block_causal_mask, max_neg_value(csim))

                importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean', block_size = self.selection_block_size)


            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]
            # handle if number of total blocks is less than number to select for fine attention

            # get the top-n kv segments for fine attention
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)

            if self.use_diff_topk:
                gates = straight_through(selected_importance_values, 1.)
                gates = gates.cumprod(dim = -1)[..., -1]
                gates = repeat(gates, 'b h ... -> b (h qh) ...', qh = fine_num_grouped_queries)


            if self.use_triton_kernel and not disable_triton_kernel:
                from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend

                fmask = selected_importance_values > 1e-10

                fine_attn_out = native_sparse_attend(
                    fq, fk, fv,
                    self.selection_block_size,
                    selected_block_indices,
                    fmask
                )

            elif self.use_fine_flex_attention:
                # flex attention for the selection for fine attention
                fn = create_fine_mask(fine_divisible_seq_len, self.selection_block_size)
                fine_block_mask = fn(selected_block_indices, fine_num_grouped_queries)

                fine_attn_out = flex_attention(fq, fk, fv, block_mask = fine_block_mask, enable_gqa = True)

            else:
                fmask = selected_importance_values > 1e-10

                if S < fine_divisible_seq_len:
                    remainder = fine_divisible_seq_len - S
                    fk = pad_at_dim(fk, (0, remainder), value = 0., dim = -2)
                    fv = pad_at_dim(fv, (0, remainder), value = 0., dim = -2)
                    fq = pad_at_dim(fq, (0, remainder), value = 0., dim = -2)

                    fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

                    if self.use_diff_topk:
                        gates = pad_at_dim(gates, (0, remainder), value = 1.)

                # handle block causal diagonal in the diagram, but run experiments without to see

                fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
                fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = B, h = selected_block_indices.shape[1])
                selected_block_indices = torch.cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

                fmask = repeat(fmask, 'b h i w -> b h i w j', j = self.selection_block_size)

                causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
                causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = B, h = fmask.shape[1])

                fmask = torch.cat((fmask, causal_mask), dim = -2)
                fmask = rearrange(fmask, 'b h i w j -> b h i (w j)')

                # select out the spatial crops of keys / values for fine attention

                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)

                if self.query_heads_share_selected_kv:
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])

                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))

                # fine attention

                fmask = rearrange(fmask, 'b h ... -> b h 1 ...')

                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)

                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.attn_scale

                mask_value = max_neg_value(fsim)

                fsim = fsim.masked_fill(~fmask, mask_value)

                fattn = fsim.softmax(dim = -1)

                fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')

                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')

                fine_attn_out = fine_attn_out[..., :S, :]

            # handle maybe gating

            if self.use_diff_topk:
                gates = gates[..., :S]
                assert self.num_heads == fine_num_grouped_queries, (
                    f"num heads must equal num grouped queries, got {self.num_heads} and {self.num_grouped_queries}"
                )
                fine_attn_out = einx.multiply(
                    'b h n, b h n d -> b h n d', 
                    gates, 
                    fine_attn_out, 
                    h = self.num_heads
                )

        else:
            # if only first block, just do a simple block causal

            seq_len = fk.shape[-2]
            fmask = causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).tril()
            fine_attn_out = F.scaled_dot_product_attention(fq, fk, fv, attn_mask = fmask)

        return fine_attn_out
        

    def forward(
        self, 
        x: Tensor, 
        ve: Tensor | None, 
        block_mask: BlockMask
    ):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        compress_divisible_seq_len = round_down_multiple(T, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_multiple(T, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        
        q = self.flip_h_seq_dim(q)
        k = self.flip_h_seq_dim(k)
        v = self.flip_h_seq_dim(v)
        
        # learnable interpolation between token value embeddings ve 
        # and current token value embeddings v
        
        # we
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
            
        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = B)
        num_mem_compress_kv = mem_ck.shape[-2]
        # step 1: compressed attention
        # NOTE compressed attention uses its own rotary embedding
        compressed_attn_out, csim = self.compressed_attention(
            q, 
            k, 
            v, 
            mem_ck, 
            mem_cv, 
            num_mem_compress_kv, 
            num_compress_blocks, 
            compress_divisible_seq_len, 
            device = x.device
        )
            
        # step 2: fine selection attention
        q, k = self.rotary(q), self.rotary(k)
        
        fine_attn_out = self.fine_attention(
            fq=q, 
            fk=k, 
            fv=v, 
            csim=csim, 
            num_mem_compress_kv=num_mem_compress_kv, 
            num_compress_blocks=num_compress_blocks, 
            num_fine_blocks=num_fine_blocks, 
            fine_divisible_seq_len=fine_divisible_seq_len,
            device=x.device
        )
        
        
        # step 3: sliding window attention
        # NOTE not tested yet
        sliding_window_attn_out = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        sliding_window_attn_out = sliding_window_attn_out.contiguous()
        
        print("x shape: ", x.shape)
        strategy_weighted_combine = self.to_strategy_combine(x)
        print("strategy_weighted_combine shape: ", strategy_weighted_combine.shape)
        
        merged_attn = torch.stack([
            compressed_attn_out, 
            fine_attn_out, 
            sliding_window_attn_out
        ], dim = 0)
        
        print("merged_attn shape: ", merged_attn.shape)
        
        out = einops.einsum(
            strategy_weighted_combine, 
            merged_attn,
            'b h n s, s b h n d -> b h n d'
        )
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        out = self.combine_heads(out) # map from 3*dim to dim
        return out
        
        


class NSABlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        layer_idx: int,
        sliding_window_size: int = 32,
        compress_block_size: int = 4,
        selection_block_size: int = 4,
        num_selected_blocks: int = 4,
    ):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = SparseAttention(
            dim=dim, 
            dim_head=dim//num_heads, 
            heads=num_heads, 
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks
        ) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(
        self, 
        x : Tensor, 
        ve : Tensor, 
        x0 : Tensor, 
        sliding_window_flex_mask: BlockMask,
        fine_selection_flex_mask: BlockMask
    ):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn.forward(
                norm(x), 
                sliding_window_flex_mask=sliding_window_flex_mask, 
                fine_selection_flex_mask=fine_selection_flex_mask
            )
        x = x + self.mlp(norm(x))
        return x


class NSA_GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        use_fp8: bool = False,
        sliding_window_size: int = 32,
        compress_block_size: int = 4,
        selection_block_size: int = 4,
        num_selected_blocks: int = 4,
        attn_fine_block_size: int = 4,
    ):
        super().__init__()
        self.use_fp8 = use_fp8
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.attn_fine_block_size = attn_fine_block_size
        self.blocks = cast(list[NSABlock], nn.ModuleList([
            NSABlock(
                dim=model_dim, 
                num_heads=num_heads, 
                layer_idx=i,
                sliding_window_size=sliding_window_size,
                compress_block_size=compress_block_size,
                selection_block_size=selection_block_size,
                num_selected_blocks=num_selected_blocks
            ) for i in range(num_layers)
        ]))
        
        
        # [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm] 
        # do reverse for decoder
        
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_() # @Grad62304977

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        long_bm, short_bm = create_block_masks(input_seq, sliding_window_num_blocks)
        fine_selection_flex_mask = create_fine_mask(input_seq, self.attn_fine_block_size)


        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks : list[BlockMask] = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].forward(
                x, 
                ve_enc[i], 
                x0, 
                sliding_window_flex_mask=block_masks[i], 
                fine_selection_flex_mask=fine_selection_flex_mask
            )
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i].forward(
                x, 
                ve_dec[i], 
                x0, 
                sliding_window_flex_mask=block_masks[i], 
                fine_selection_flex_mask=fine_selection_flex_mask
            )
        x = norm(x)
        
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training and self.use_fp8 else self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss
