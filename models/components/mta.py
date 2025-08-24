# implementation of multi-token attention

import torch
import torch.nn as nn
from jaxtyping import Float, Int
import math
import torch.nn.functional as F
import einops
import json

class MTA(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        dropout=0.1, 
        attn_weight_kernel_size=3, 
        head_kernel_size=3
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.Q = nn.Linear(d_model, d_model*num_heads)
        self.K = nn.Linear(d_model, d_model*num_heads)
        self.V = nn.Linear(d_model, d_model*num_heads)
        self.O = nn.Linear(d_model*num_heads, d_model)
        self.attn_weight_mixer = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=attn_weight_kernel_size, padding=attn_weight_kernel_size//2)
        self.head_mixer = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=head_kernel_size, padding=head_kernel_size//2)

    def forward(
        self, 
        x: Float[torch.Tensor, "batch seq_len d_model"],
        mask: Int[torch.Tensor, "seq_len seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:

        B = x.shape[0]
        S = x.shape[1]
        H = self.num_heads
        D = self.d_model
        Q = self.Q(x).view(B, H, S, D)
        K = self.K(x).view(B, H, S, D)
        V = self.V(x).view(B, H, S, D)
        attn_logits = einops.einsum(Q, K, "b h s1 d, b h s2 d -> b h s1 s2") / math.sqrt(D)
        attn_logits *= mask # mask out the padding tokens
        attn_logits = self.attn_weight_mixer(attn_logits)
        attn_logits = mask * self.head_mixer(attn_logits) # we apply the mask twice
        attn_weights = F.softmax(attn_logits, dim=-1)
        out = attn_weights @ V
        out = out.view(B, S, -1)
        out = self.O(out)
        return out
