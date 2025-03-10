from typing import Dict
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
if torch.cuda.is_available():
    from ops import lm_head_fp8
else:
    lm_head_fp8 = None
from utils import next_multiple_of_n
from models.shared import ValueEmbedding, Block, CastedLinear, norm
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

    def forward(
        self, 
        x: Tensor, 
        x0: Tensor, 
        block_mask: BlockMask, 
        targets: Tensor,
        with_backward: bool = False
    ):
        """
        x: (B, T, H)
        next_token_embs: (B, N, H) where N is n_tokens
        x0: (B, T, H)
        block_mask: (B, T)
        targets: (B, T+N)
        """
        if x.ndim== 3:
            # we flattent to one batch sequence 
            # as flexattention expects a single sequence 
            x = x.view(1, -1, x.shape[-1])
            x0 = x0.view(1,-1, x0.shape[-1])
            
        targets = targets.view(-1)    
        assert x.shape[1] == targets.shape[0] - self.n_tokens

        
        current_hidden = x
        T = x.shape[1]
        
        loss_dict = {}
        loss_accum = 0
        for i in range(self.n_tokens):
            # Create a fresh computation graph for each iteration
            if with_backward:
                current_hidden = current_hidden.detach()
                current_hidden.requires_grad = True
                current_hidden.retain_grad()
            
            # Normalize inputs
            embs = x0[:, i:i+T, :]
            
            x_norm = norm(current_hidden)
            next_norm = norm(embs)
            
            # Combine and project
            combined = torch.cat([x_norm, next_norm], dim=-1)
            h_prime = self.projs[i].forward(combined)
            
            # Pass through transformer block
            block : Block = self.blocks[i]
            current_hidden = block.forward(h_prime, None, embs, block_mask)
            
            # Get predictions and compute loss
            shift_targets = targets[i:i+T].contiguous().view(-1)
            head = self.shared_head
            
            if self.use_liger:
                loss = self.loss_fn.forward(
                    head.weight.to(current_hidden.dtype), 
                    current_hidden.view(-1, current_hidden.shape[-1]),
                    shift_targets,
                    bias=head.bias,
                )
            else:
                if self.proj_fp8:
                    logits = lm_head_fp8(current_hidden, head.weight)
                else:
                    logits = head.forward(current_hidden)
                loss = self.loss_fn.forward(logits.view(-1, self.d_vocab), shift_targets)
            
            loss_dict[f"loss_{'orig' if i==0 else 'token_' + str(i)}"] = loss.detach()
            
            if with_backward:
                # Compute gradients for this iteration
                loss.backward(retain_graph=True)
                loss_accum += loss.detach()
                
                # Accumulate gradients for shared_trunk using autograd.grad with allow_unused=True
                grads = torch.autograd.grad(loss, x, retain_graph=True, allow_unused=True)[0]
                if grads is not None:  # Only accumulate if we got gradients
                    if x.grad is None:
                        x.grad = grads
                    else:
                        x.grad += grads
            else:
                loss_accum += loss
        
        return loss_accum, loss_dict
    



class MTP(nn.Module):
    def __init__(
        self, 
        d_hidden: int, 
        n_tokens: int, 
        d_vocab: int,
        use_liger: bool = False,
        proj_fp8: bool = False
    ):
        super().__init__()
        self.n_tokens = n_tokens
        
        # Multiple heads now output logits for classification
        self.heads = nn.ModuleList([
            CastedLinear(d_hidden, d_vocab) for _ in range(n_tokens)
        ])
        
        self.use_liger = use_liger
        self.proj_fp8 = proj_fp8
        if use_liger:
            self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(
        self, 
        x, 
        targets,
        with_backward: bool = False
    )-> Tensor | Dict[str, Tensor]:
        """
        Implementation with CrossEntropyLoss for sequence prediction
        x: shape [batch_size, seq_len, input_dim]
        targets: shape [batch_size, seq_len + n_tokens]
        """

        if x.ndim==3:
            BATCH_SIZE = x.shape[0]
            x = x.view(-1, x.shape[-1])
        
        if targets.ndim==2:
            assert targets.shape[0] == BATCH_SIZE
            targets = targets.view(-1)
            
        assert x.shape[0] == targets.shape[0] - (BATCH_SIZE*self.n_tokens)

        z = x  # [batch_size, seq_len, hidden_dim]

        if with_backward:
            # Step 2: Detach and enable grad tracking
            d = z.detach()
            d.requires_grad = True
        else:
            d = z
            
        # Step 3: Sequential forward/backward through heads
        total_loss = 0


        loss_dict = {}
        T = x.shape[0] # seq_len
        for i in range(self.n_tokens):
            # Get shifted labels for this head
            
            head_labels = targets[..., i:T+i] 
            head_labels = head_labels.contiguous().view(-1)
            
            if self.use_liger:
                bias = self.heads[i].bias
                if bias is not None:
                    bias = bias.to(d.dtype)
                    
                loss = self.loss_fn.forward(
                    self.heads[i].weight.to(d.dtype),
                    d,
                    head_labels,
                    bias=bias,
                )
            else:
                # Forward through head
                if self.proj_fp8:
                    logits = lm_head_fp8(d, self.heads[i].weight)
                else:
                    logits = self.heads[i](d)  # [batch_size, seq_len, num_classes]
                # Reshape logits to match labels
                logits = logits.view(-1, logits.size(-1))
                loss = self.loss_fn(logits, head_labels)
            total_loss += loss
            loss_dict[f"loss_{'orig' if i==0 else 'token_' + str(i)}"] = loss.detach()
            
            if with_backward:
                # Backward for this head
                loss.backward()

        if with_backward:
            # Step 4: Backward through shared layer with accumulated gradients
            z.backward(d.grad)
        
        return total_loss, loss_dict


class MTPGPT(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        num_layers: int, 
        num_heads: int, 
        model_dim: int, 
        n_mtp_tokens: int, 
        use_deepseek_mtp: bool = False,
        use_liger: bool = True,
        proj_fp8: bool = True,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.device = device
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
                num_heads=num_heads,
                d_hidden=model_dim,
                d_vocab=next_multiple_of_n(vocab_size, n=128),
                use_liger=use_liger,
                proj_fp8=proj_fp8
            )
        else:
            self.mtp = MTP(
                d_hidden=model_dim,
                n_tokens=n_mtp_tokens,
                d_vocab=next_multiple_of_n(vocab_size, n=128),
                use_liger=use_liger,
                proj_fp8=proj_fp8
            )
            
        print(f"Using {self.mtp.__class__.__name__} for MTP")
            
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
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=self.device)
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
        sliding_window_num_blocks: Tensor,
        with_backward: bool = False
    )-> Tensor | tuple[Tensor, Tensor]:
        assert input_seq.ndim == 1

        # we process the input sequence without the last mtp.n_tokens
        # as usual.
        k_excl_input_seq = input_seq[..., :-self.mtp.n_tokens] # exclude the last mtp.n_tokens
        long_bm, short_bm = self.create_block_masks(k_excl_input_seq, sliding_window_num_blocks)
        x0 = norm(self.embed(input_seq)[None]) 
        # we need the embeddings for the last mtp.n_tokens also
        # Get embeddings for the target sequence tokens
        
        # NOTE does this allocate new memory?
        sliced_x = x0_for_processing = x0[:, :-self.mtp.n_tokens, :]
        ve = self.value_embeds(k_excl_input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers


        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm]
        for i in range(self.num_encoder_layers):
            sliced_x = self.blocks[i](sliced_x, ve_enc[i], x0_for_processing, block_masks[i])
            skip_connections.append(sliced_x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        for i in range(self.num_decoder_layers):
            sliced_x = sliced_x + self.skip_weights[i] * skip_connections.pop()
            sliced_x = self.blocks[self.num_encoder_layers + i](sliced_x, ve_dec[i], x0_for_processing, block_masks[i])
        sliced_x = norm(sliced_x)
        

        if isinstance(self.mtp, DeepSeekV3MTP):
            loss, loss_dict = self.mtp.forward(
                x=sliced_x,
                x0=x0,
                block_mask=long_bm,
                targets=target_seq,
                with_backward=False# NOTE it doesnt currently work with_backward
            )
            if with_backward:
                loss.backward() # NOTE only to test how much slower it is
                
            return loss, loss_dict
        else:
            loss = self.mtp.forward(
                x=sliced_x,
                targets=target_seq,
                with_backward=with_backward
            )
        
        return loss

            