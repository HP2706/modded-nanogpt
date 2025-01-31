import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from ops import lm_head_fp8
from utils import next_multiple_of_n
from models.shared import ValueEmbedding, Block, CausalSelfAttention, MLP, CastedLinear, norm
from models.Current_best_gpt import GPT
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


#deepseek v3 Multi token prediction model
# each prediction head is a transformer block with a linear projection
# 
class DeepSeekV3MTP(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_hidden: int,
        d_vocab: int,
        num_heads: int,
        use_liger: bool = True,
        proj_fp8: bool = True
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_hidden = d_hidden
        self.d_vocab = d_vocab
        self.use_liger = use_liger
        self.proj_fp8 = proj_fp8
        
        # Create prediction blocks for each token
        self.blocks = nn.ModuleList([
            Block(d_hidden, num_heads, layer_idx) for layer_idx in range(n_tokens)
        ])
        # Projection layers for each prediction head
        self.projs = nn.ModuleList([
            CastedLinear(d_hidden * 2, d_hidden) for _ in range(n_tokens)
        ])
        
        # Output heads
        self.shared_head = CastedLinear(d_hidden, d_vocab)
        self.shared_head.weight.detach().zero_()
            
        self.use_liger = use_liger
        self.proj_fp8 = proj_fp8
        if use_liger:
            self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, hidden_states: Tensor, next_token_embs: Tensor, x0: Tensor, block_mask, labels: Tensor):
        """
        hidden_states: (B, T, H)
        next_token_embs: (B, N, H) where N is n_tokens
        labels: (B, T+N)
        x0: (B, T, H)
        """
        assert hidden_states.shape[1] == labels.shape[1] - self.n_tokens
        assert next_token_embs.shape[1] == self.n_tokens
        
        shared_trunk = hidden_states.detach()
        shared_trunk.requires_grad = True
        
        loss_accum = 0
        for i in range(self.n_tokens):
            # Normalize inputs
            x_norm = norm(shared_trunk)
            next_norm = norm(next_token_embs[:, i])
            
            # Combine and project
            combined = torch.cat([x_norm, next_norm], dim=-1)
            h_prime = self.projs[i].forward(combined)
            
            # Pass through transformer block
            block_output = self.blocks[i].forward(h_prime, None, x0, block_mask)
            
            # Get predictions and compute loss
            shift_labels = labels[..., (1 + i):].contiguous().view(-1)
            head = self.shared_head
            
            if self.use_liger:
                loss = self.loss_fn.forward(
                    head.weight,
                    block_output,
                    shift_labels,
                    bias=head.bias,
                )
            else:
                if self.proj_fp8:
                    logits = lm_head_fp8(block_output, head.weight)
                else:
                    logits = head.forward(block_output)
                loss = self.loss_fn.forward(logits.view(-1, self.d_vocab), shift_labels)
            
            loss.backward()
            loss_accum += loss.detach()
            
        return loss_accum, shared_trunk.grad


class MTP(nn.Module):
    def __init__(
        self, 
        n_tokens : int,
        d_hidden : int,
        d_vocab : int,
        use_liger : bool = True,
        proj_fp8 : bool = True
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_hidden = d_hidden
        self.d_vocab = d_vocab
        self.heads = torch.nn.ModuleList([
            CastedLinear(d_hidden, d_vocab) for _ in range(n_tokens)
        ])
        for head in self.heads:
            head.weight.detach().zero_() # @Grad62304977
        
        self.use_liger = use_liger
        self.proj_fp8 = proj_fp8
        
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
                if self.proj_fp8:
                    logits = lm_head_fp8(shared_trunk, head.weight)
                else:
                    logits = head.forward(shared_trunk)
                
                loss = self.loss_fn.forward(logits.view(-1, self.d_vocab), shift_labels)
            
            #NOTE we call backward multiple times to accumulate the gradients to d(the shared trunk)
            loss.backward()
            loss_accum += loss.detach() # We only need this for logging
            
        return loss_accum, shared_trunk.grad

class MTPGPT(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        num_layers: int, 
        num_heads: int, 
        model_dim: int, 
        n_mtp_tokens: int, 
        use_deepseek_mtp: bool = True,
        use_liger: bool = True,
        proj_fp8: bool = True
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
        
        if use_deepseek_mtp:
            self.mtp = DeepSeekV3MTP(
                n_tokens=n_mtp_tokens,
                d_hidden=model_dim,
                d_vocab=next_multiple_of_n(vocab_size, n=128),
                use_liger=use_liger,
                proj_fp8=proj_fp8
            )
        else:
            self.mtp = MTP(
                n_tokens=n_mtp_tokens,
                d_hidden=model_dim,
                d_vocab=next_multiple_of_n(vocab_size, n=128),
                use_liger=use_liger,
                proj_fp8=proj_fp8
            )
            
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
        
        # we backpropagate the loss by doing x.backward(gradient=shared_trunk_grad)
        # instead of loss.backward()
        # we keep the loss for logging
        loss, shared_trunk_grad = self.mtp.forward(x, target_seq)
        return loss, x, shared_trunk_grad 
            