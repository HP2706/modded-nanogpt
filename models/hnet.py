# implementation of H-Net https://arxiv.org/pdf/2507.07955
#from shared import *
from typing import Tuple, Optional, List, cast, Dict
import einops
from torch import nn
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
import math
from models.shared import Block, norm
from models.Current_best_gpt import create_block_masks

# https://github.com/state-spaces/mamba/releases/tag/v2.2.5 get wheels

class RoutingModuleOutput(BaseModel):
    boundary_probs : torch.Tensor
    boundary_mask : torch.Tensor
    selection_mask : torch.Tensor
    
    class Config:
        arbitrary_types_allowed = True

def smoothing_fn(
    z : Float[torch.Tensor, "B L D"],
    boundary_probs : Float[torch.Tensor, "B L 2"],
):
    return F.pad(
        torch.sum(
            z[:, 1:] * boundary_probs[:, :, 1] + 
            z[:, :-1] * boundary_probs[:, :, 0], dim=-1), 
        (1, 0), value=0)


class RoutingModule(nn.Module):
    def __init__(self, d_in : int):
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_in)
        self.W_K = nn.Linear(d_in, d_in)
        
    def ratio_loss(self, 
        boundary_probs : torch.Tensor, 
        selection_mask : torch.Tensor,
        target_compression_ratio : float,
    ):
        '''
        the special loss function eq 10
        
        Minimized when F=G=1/N with minimum being 1.0
        '''
        G = boundary_probs.mean(dim=-1) # the mean of the boundary probabilities
        F = selection_mask.float().mean(dim=-1) # the fraction of selected positions 
        N = target_compression_ratio * selection_mask.shape[-1] # the targeted number of selected positions
        L_ratio = (N / (N-1)) * ((N-1)*F*G+(1-G)*(1-F))
        return L_ratio
        
    def forward(
        self, 
        x : Float[torch.Tensor, "B L D"], 
        mask : Bool[torch.Tensor, "B L"]
    ) -> RoutingModuleOutput:
        '''
        Routing Module eq 4 in the paper
        Computes boundary probabilities p_t between adjacent positions
        
        compute pairwise cosine similarity between adjacent positions
        '''
        Q = self.W_Q(x)[:, :-1]
        K = self.W_K(x)[:, 1:]
        
        cos_sim = einops.einsum(F.normalize(Q, dim=-1), F.normalize(K, dim=-1), 'b l d, b l d -> b l') # compute pairwise dot product # (B, L-1)
        A = torch.clamp((1/2) * (1 - cos_sim), min=0, max=1)
        A = F.pad(A, (1, 0), value=1) 
        # pad with 1 at beginning as it by definition is a boundary per the paper
        
        boundary_probs = torch.stack(((1 - A), A), dim=-1) # [B, L, 2]
        boundary_mask = torch.argmax(boundary_probs, dim=-1) # [B, L]
        selected_boundaries = torch.argmax(boundary_probs, dim=-1) # [B, L]
        boundary_mask = boundary_mask & mask
        
        return RoutingModuleOutput(
            boundary_probs=boundary_probs, 
            boundary_mask=boundary_mask, 
            selection_mask=selected_boundaries)
        
def down_sample(
    hidden_states : Float[torch.Tensor, "B L D"], 
    boundary_mask : Bool[torch.Tensor, "B L"],
) -> Tuple[
    Float[torch.Tensor, "B L D"],
    Bool[torch.Tensor, "B L"],
]:
    '''
    '''
    device = hidden_states.device
    L = hidden_states.shape[1]
    num_tokens = boundary_mask.sum(dim=-1)
    next_max_seqlen = int(num_tokens.max())

    device = hidden_states.device
    L = hidden_states.shape[1]
    flat_indices = (
        torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
    ) # [B, L]
    seq_sorted_indices = torch.argsort(flat_indices, dim=1)

    next_hidden_states = torch.gather(
        hidden_states,
        dim=1,
        index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
            -1, -1, hidden_states.shape[-1]
        ),
    )

    next_mask = (
        torch.arange(next_max_seqlen, device=device)[None, :]
        < num_tokens[:, None]
    )
    next_max_seqlen = None

    return next_hidden_states, next_mask

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x : torch.Tensor) -> torch.Tensor:
    return STE.apply(x)

def upsample(
    z : Float[torch.Tensor, 'B L D'],
    boundary_probs : Float[torch.Tensor, 'B L 2'],
    boundary_mask : Bool[torch.Tensor, 'B L']
) -> torch.Tensor:
    '''
    Upsample the compressed hidden states z to the original sequence length
    
    Implements equations 8 and 9:
    - eq 8: z̃_t = z_{∑_{k=1}^t b_k}  (causal expansion)
    - eq 9: Upsampler(z̃, c)_t = STE(c_t) · z̃_t  (confidence-weighted decompression)
    '''
    B, L_compressed, D = z.shape
    B, L_original = boundary_mask.shape
    
    # Step 1: Compute confidence scores (equation 6 from paper)
    # c_t = p_t^{b_t} * (1-p_t)^{1-b_t}
    # This gives high confidence when boundary predictions are correct
    confidence = (boundary_probs[:, :, 1] ** boundary_mask.float()) * (boundary_probs[:, :, 0] ** (1 - boundary_mask.float()))
    
    # Step 2: Apply STE (Straight-Through Estimator) for gradient stabilization
    confidence_ste = ste_func(confidence)
    
    # Step 3: Compute cumulative boundary indices (equation 8)
    # ∑_{k=1}^t b_k gives the index into the compressed sequence z
    cumulative_boundaries = torch.cumsum(boundary_mask.float(), dim=-1)
    
    # Clamp to valid indices (boundary_mask should ensure this, but safety first)
    cumulative_boundaries = torch.clamp(cumulative_boundaries - 1, min=0, max=L_compressed - 1).long()
    
    # Step 4: Expand compressed sequence to original length (equation 8)
    # z̃_t = z_{∑_{k=1}^t b_k}
    z_expanded = torch.gather(
        z, 
        dim=1, 
        index=cumulative_boundaries.unsqueeze(-1).expand(-1, -1, D)
    )

    # Step 5: Apply confidence weighting (equation 9)
    # Upsampler(z̃, c)_t = STE(c_t) · z̃_t
    upsampled = confidence_ste.unsqueeze(-1) * z_expanded
    
    return upsampled

class EncoderOutput(BaseModel):
    hidden_states : Float[torch.Tensor, "B L D"]
    encoder_outputs : List[Float[torch.Tensor, "B L D"]]  # Add encoder outputs for residual connections
    loss : Optional[torch.Tensor] = None
    
    class Config:
        arbitrary_types_allowed = True
    

class Encoder(nn.Module):
    def __init__(
        self, 
        d_model : int, 
        compression_ratio : float, 
        n_layers : int,
        num_heads : int,
    ):
        super().__init__()  # Add missing super().__init__()
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.n_layers = n_layers
        self.chunking_modules = nn.ModuleList([
            RoutingModule(d_model) for _ in range(n_layers)
        ])
        
        #TODO use mamba blocks instead of blocks
        self.blocks = nn.ModuleList([Block(d_model, num_heads, layer_idx, dummy=True) for layer_idx in range(n_layers)])
        
        # NOTE we assume we target the same compression ratio for each layer
        # which corresponds to solving 
        # compression_ratio = a^n_layers => a =exp(log(compression_ratio) / n_layers)
        # NOTE in the future this could be learned, ie we should let layers decide
        # how much each compress as long as the total compression ratio is met after n_layers
        self.target_compression_ratio = math.exp(math.log(compression_ratio) / n_layers)
        
    def forward(
        self,
        hidden_states : Float[torch.Tensor, "B L D"],
        causal_mask : Bool[torch.Tensor, "B L"],
    ) -> EncoderOutput:
        losses = []
        encoder_outputs = []  # Store intermediate outputs for residual connections
        
        next_causal_mask = causal_mask
        for i, (chunking_module, block) in enumerate(zip(
            cast(List[RoutingModule], self.chunking_modules),
            cast(List[Block], self.blocks)
        )):
            # Apply block processing first
            hidden_states = norm(hidden_states)
            hidden_states = block.forward(
                hidden_states, 
                ve=hidden_states, 
                x0=hidden_states, 
                block_mask=next_causal_mask, 
            )
            
            # Store the output before downsampling for residual connections
            encoder_outputs.append(hidden_states.clone())
            
            # Apply routing and downsampling
            output = chunking_module.forward(hidden_states, next_causal_mask)
            
            
            
            hidden_states, next_causal_mask = down_sample(
                hidden_states, output.boundary_mask)
            
            
            losses.append(chunking_module.ratio_loss(
                output.boundary_probs, 
                output.selection_mask, 
                self.target_compression_ratio
            ))

        return EncoderOutput(
            hidden_states=hidden_states, 
            encoder_outputs=encoder_outputs,  # Add encoder outputs for residual connections
        )
        

class Decoder(nn.Module):
    def __init__(
        self, 
        d_model : int, 
        compression_ratio : float,  
        n_layers : int,
        num_heads : int,
    ):
        super().__init__()  # Add missing super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.decompression_ratio = 1 / compression_ratio # go the other way
        self.target_decompression_ratio = math.exp(math.log(self.decompression_ratio) / n_layers)
        
        self.blocks = nn.ModuleList([Block(d_model, num_heads, layer_idx, dummy=True) for layer_idx in range(n_layers)])
        
        self.routing_modules = nn.ModuleList([
            RoutingModule(d_model) for _ in range(n_layers)
        ])
        
        # Add learnable weights for residual connections
        # These weights control the strength of each residual connection
        self.residual_weights = nn.Parameter(torch.ones(n_layers))
        
    def forward(
        self,
        hidden_states : Float[torch.Tensor, "B L D"],
        causal_mask : Bool[torch.Tensor, "B L"],
        encoder_outputs : List[Float[torch.Tensor, "B L D"]],  # Add encoder outputs parameter
    ) -> Float[torch.Tensor, "B L D"]:
        
        x0 = hidden_states
        ve = hidden_states #TODO we need to carry information from each compression / expansion
        
        # Reverse encoder outputs and routing info to match decoder layer order
        # encoder_outputs[0] corresponds to the deepest encoder layer
        # decoder layer 0 (first) should connect to encoder layer n_layers-1 (last)
        encoder_outputs_reversed = list(reversed(encoder_outputs))
        
        for i, (routing_module, block) in enumerate(zip(
            cast(List[RoutingModule], self.routing_modules),
            cast(List[Block], self.blocks)
        )):
            # Apply residual connection from corresponding encoder layer
            # We'll upsample the encoder residual later with the current layer's routing info
            # Apply block processing
            hidden_states = norm(hidden_states)
            hidden_states = block.forward(
                hidden_states, 
                ve=ve, 
                x0=x0, 
                block_mask=causal_mask, 
            )
            
            # Apply routing and upsampling
            output = routing_module.forward(hidden_states, causal_mask)
            
            hidden_states = upsample(
                hidden_states, output.boundary_probs, output.boundary_mask
            )
            
            # Apply weighted residual connection
            # Upsample encoder_residual to match the current hidden_states size
            encoder_residual_upsampled = upsample(
                encoder_outputs_reversed[i], 
                output.boundary_probs, 
                output.boundary_mask
            )
            
            hidden_states = hidden_states + self.residual_weights[i] * encoder_residual_upsampled
            
        return hidden_states
    
class TransformerPP(nn.Module):
    def __init__(
        self,
        model_dim : int,
        num_heads : int,
        num_layers : int,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx, dummy=True) for layer_idx in range(num_layers)])
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        
    def forward(
        self,
        hidden_states : Float[torch.Tensor, "B L D"],
        causal_mask : Bool[torch.Tensor, "B L"],
        sliding_window_num_blocks : Int[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B L D"]:
        assert hidden_states.ndim == 2, \
            "hidden_states must be 2D (Num_tokens, D_model)"

        long_bm, short_bm = create_block_masks(hidden_states, sliding_window_num_blocks)
        ve = x0 = x = hidden_states
        
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
        return x
    
class HNet(nn.Module):
    def __init__(
        self, 
        d_model : int, 
        num_heads : int,
        num_compression_layers : int,
        compression_ratio : float, 
        num_main_layers : int,
    ):
        super().__init__()  # Add missing super().__init__()
        self.embed = nn.Embedding(8, d_model)
        self.lm_head = nn.Linear(d_model, 8)
        
        self.encoder = Encoder(d_model, compression_ratio, num_compression_layers, num_heads)
        self.unembed = nn.Linear(d_model, 8) # 8 is bytes
        self.main_model = TransformerPP(d_model, num_heads, num_main_layers)
        self.decoder = Decoder(d_model, compression_ratio, num_compression_layers, num_heads)
        
    def forward(
        self,
        input_seq : Int[torch.Tensor, "B L"],
        causal_mask : Bool[torch.Tensor, "B L"],
        sliding_window_num_blocks : Int[torch.Tensor, "B"],
    ) -> Float[torch.Tensor, "B L D"]:
        
        hidden_states = self.embed(input_seq)
        
        # Pass through encoder and get intermediate outputs
        encoder_output = self.encoder.forward(hidden_states, causal_mask)
        
        # Pass through main model
        hidden_states = self.main_model(encoder_output.hidden_states, causal_mask, sliding_window_num_blocks)
        
        # Pass through decoder with residual connections
        hidden_states = self.decoder.forward(
            hidden_states, 
            causal_mask, 
            encoder_output.encoder_outputs,  # Pass encoder outputs for residual connections
        )
        
        return self.lm_head(hidden_states)