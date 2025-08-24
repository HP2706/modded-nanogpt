import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math
from models.components.shared import CausalSelfAttention, CausalSelfAttentionPlain, MLP, norm, Block, CastedLinear, ValueEmbedding, create_block_masks
from ops import lm_head_fp8
from utils import next_multiple_of_n
from typing import Literal
from torch import Tensor
"""
A combination of HNet and NSA.
where we at each encoder step use mlp to compress consecutive tokens
"""

def maybe_pad_to_multiple_of_n(x : torch.Tensor, n : int, dim : int = 1) -> torch.Tensor:
    assert x.ndim == 3, \
        f"Input must be a 3D tensor, got {x.ndim}D"
        
    print("x.shape[1]", x.shape[1], "n", n, "x.shape[1] % n", x.shape[1] % n)
    if x.shape[1] % n != 0:
        padding_needed = n - x.shape[1] % n
        print("padding x", x.shape, "to sequence length", x.shape[1] + padding_needed)
        # Pad the sequence length dimension (dimension 1)
        new_x = torch.nn.functional.pad(x, (0, 0, 0, padding_needed))
        print("new_x", new_x.shape)
        return new_x
    return x

class CompressMLP(nn.Module):
    def __init__(self, 
        dim : int, 
        expand_or_reduce_factor : int = 1,
        mode : Literal["compress", "decompress"] = "compress",
    ):
        super().__init__()
        assert mode in ["compress", "decompress"], \
            f"Mode must be either 'compress' or 'decompress', got {mode}"

        self.expand_or_reduce_factor = expand_or_reduce_factor
        self.mode = mode

        if mode == "compress":
            d_in = dim * expand_or_reduce_factor
            d_out = dim
        else:
            d_in = dim
            d_out = dim * expand_or_reduce_factor
            
        d_hidden = 4 * d_in
        self.W_1 = nn.Linear(d_in, d_hidden)
        self.W_2 = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        # if integer then we are expandin
        
        x = maybe_pad_to_multiple_of_n(x, self.expand_or_reduce_factor)
        
        if self.mode == "compress":
            x = einops.rearrange(x, 'b (h l) d -> b h (l d)', l=self.expand_or_reduce_factor)

        x = self.W_1(x)
        x = self.relu(x)
        x = self.W_2(x)

        if self.mode == "decompress":
            x = einops.rearrange(x, 'b h (l d) -> b (h l) d', l=self.expand_or_reduce_factor)

        # fold back to original shape now with shorter sequence length
        assert x.shape[-1] == d, \
            f"Input and output dimensions must match, got {x.shape[-1]} and {d}"
        return x

class CompressBlock(nn.Module):
    def __init__(self, 
        dim : int, 
        expand_or_reduce_factor : int,
        num_heads : int,
        layer_idx : int,
        mode : Literal["compress", "decompress"] = "compress",
        dummy : bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.compress_mlp = CompressMLP(dim, expand_or_reduce_factor, mode)
        
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        if dummy:
            self.attn = CausalSelfAttentionPlain(dim, num_heads, head_dim=dim//num_heads, layer_idx=layer_idx)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, layer_idx, head_dim=dim//num_heads) if layer_idx != 7 else None
        
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        x_processed = self.compress_mlp(x)
        #x0_processed = self.compress_mlp(x0)
        
        #x_processed = self.lambdas[0] * x_processed + self.lambdas[1] * x0_processed
        
        if self.attn is not None:
            x_processed = x_processed + self.attn.forward(norm(x_processed), ve, block_mask)
        x_processed = x_processed + self.mlp(norm(x_processed))
        
        # Return original input for compression mode, processed output for decompression mode
        return x_processed
        
        
        
class HNetXNSA(nn.Module):
    def __init__(self, 
        vocab_size: int,
        num_heads: int,
        model_dim: int,
        n_inner_layers: int,
        n_compress_decompress_layers: int,
        compression_decompress_size: int,
        use_fp8: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.use_fp8 = use_fp8
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.n_total_layers = n_inner_layers + 2*n_compress_decompress_layers 

        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(self.n_total_layers)
        ])
        
        compress_per_layer = math.exp(math.log(compression_decompress_size) / n_compress_decompress_layers)
        assert compress_per_layer.is_integer(), "Compression/decompression size must be a power of 2"
        compress_per_layer = int(compress_per_layer)
        
        print("compress_per_layer", compress_per_layer)
        self.encoder_layers = nn.ModuleList([
            CompressBlock(
                model_dim, compress_per_layer, num_heads, 
                layer_idx, 
                mode="compress", 
                dummy=dummy
            )
            for layer_idx in range(n_compress_decompress_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            CompressBlock(
                model_dim, compress_per_layer, num_heads, 
                layer_idx, 
                mode="decompress", 
                dummy=dummy
            )
            for layer_idx in range(n_compress_decompress_layers)
        ])
        
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, layer_idx, dummy=dummy) 
            for layer_idx in range(n_inner_layers)
        ])
        
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        self.lm_head.weight.detach().zero_() # @Grad62304977

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        #assert input_seq.ndim == 1

        #long_bm, short_bm = create_block_masks(input_seq, sliding_window_num_blocks)
        long_bm = sliding_window_num_blocks
        
        x = x0 = norm(self.embed(input_seq)) # use of norm here by @Grad62304977
        ve = [self.value_embeds[i](input_seq) for i in range(self.n_total_layers)]
        assert len(ve) == self.n_total_layers
        
        # Split value embeddings for encoder, blocks, and decoder
        ve_enc = ve[:len(self.encoder_layers)]
        ve_blocks = ve[len(self.encoder_layers):len(self.encoder_layers) + len(self.blocks)]
        ve_dec = ve[len(self.encoder_layers) + len(self.blocks):]
        
        assert len(ve_enc) == len(self.encoder_layers) and len(ve_blocks) == len(self.blocks) and len(ve_dec) == len(self.decoder_layers)

        # Encoder pass
        block_masks = [long_bm] * self.n_total_layers #type: ignore only use long bm for now
        #[long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(len(self.encoder_layers)):
            ve_enc[i] = x # TODO fix this find a way to expand to match sequence dimension
            x0 = x
            x = self.encoder_layers[i](x, ve_enc[i], x0, block_masks[i % len(block_masks)])
        
        # Main blocks pass
        for i in range(len(self.blocks)):
            ve_blocks[i] = x # TODO fix this find a way to expand to match sequence dimension
            x0 = x
            x = self.blocks[i](x, ve_blocks[i], x0, block_masks[i % len(block_masks)])
        
        # Decoder pass
        block_masks.reverse()
        for i in range(len(self.decoder_layers)):
            ve_dec[i] = x # TODO fix this find a way to expand to match sequence dimension
            x0 = x
            x = self.decoder_layers[i](x, ve_dec[i], x0, block_masks[i % len(block_masks)])
        
        x = norm(x)
        
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training and self.use_fp8 else self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
        return loss