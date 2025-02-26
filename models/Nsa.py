from typing import cast
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from utils import next_multiple_of_n
from native_sparse_attention_pytorch import SparseAttention
from ops import lm_head_fp8
from models.shared import ValueEmbedding, norm, CastedLinear, MLP, create_block_masks
from torch.nn.attention.flex_attention import BlockMask

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

    def forward(self, x : Tensor, ve : Tensor, x0 : Tensor, block_mask : BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn.forward(norm(x), sliding_window_flex_mask=None, fine_selection_flex_mask=None)
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
    ):
        super().__init__()
        self.use_fp8 = use_fp8
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        
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
            x = self.blocks[i].forward(x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_masks[i])
        x = norm(x)
        
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training and self.use_fp8 else self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss
