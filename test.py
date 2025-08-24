from dataclasses import dataclass
import torch
#from models.hnet import HNet, Encoder, Decoder
from models.components.mta import MTA
from models.hnet_x_nsa import CompressBlock, CompressMLP, HNetXNSA
#from models.hnet_ref.modules.dc import DeChunkState
from models.hnet import causal_conv
from torch import nn

@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)

class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        # Just for Mamba2 kernel.
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is not provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = causal_conv(x, p)
        print(out.shape)

        if cu_seqlens is not None:
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1
            out = torch.gather(
                out, dim=0, index=plug_back_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
            out = torch.gather(
                out,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        # B_selected = hidden_states.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)



if __name__ == "__main__":
    """ model = HNetXNSA(
        vocab_size = 16,
        num_heads = 1,
        model_dim = 64,
        n_inner_layers = 1,
        n_compress_decompress_layers = 2,
        compression_decompress_size = 16,
        use_fp8 = False,
        dummy = True,
    )
    
    input_seq = torch.randint(0, 16, (1, 256))
    target_seq = torch.randint(0, 16, (1, 256))
    sliding_window_num_blocks = torch.ones(1, 100)
    
    print("input_seq", input_seq.shape)
    model.forward(
        input_seq = input_seq,
        target_seq = target_seq,
        sliding_window_num_blocks = sliding_window_num_blocks,
    )
     """
     
     

    # sample probability
    boundary_prob_single = torch.rand(1, 100)
    boundary_probs = torch.stack([boundary_prob_single, 1 - boundary_prob_single], dim = -1)
    
    boundary_mask = boundary_probs[:, :, 0] > 0.5
    hidden_states = torch.randn(1, 100, 64)
    
    dechunk_layer = DeChunkLayer(
        d_model = 64,
        dtype = torch.bfloat16,
        block_size = 256,
        headdim = 32,
    )
    
    mask = torch.ones(1, 100)
    
    o = dechunk_layer.forward(hidden_states, boundary_mask, boundary_probs, mask = mask)
    print(o.shape)