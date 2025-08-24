from typing import Callable, Optional, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped, Int, Bool
from typing import OrderedDict
from Models.LLMS.configs import BaseTransformerConfig, BaseTransformerConfig, MambaConfig
try:
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
except ImportError:
    layer_norm_fn, rms_norm_fn = None, None

class MLP(nn.Module):
    def __init__(self, cfg : BaseTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_model*cfg.d_mult)
        self.fc2 = nn.Linear(cfg.d_model*cfg.d_mult, cfg.d_model)
        self.fc2.NANOGPT_SCALE_INIT = 1 #type: ignore
        self.nonlin = cfg.nonlin

    ##@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Union[
            Float[Tensor, "batch sequence_len d_model"],
            Float[Tensor, "batch d_model"]
        ],
    )-> Union[
        Float[Tensor, "batch sequence_len d_model"],
        Float[Tensor, "batch d_model"]
    ]:
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        return x

class UnEmbedding(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Linear(cfg.d_model, cfg.vocab_size, bias = False)

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators
    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"],
    )-> Float[Tensor, "batch sequence_len vocab_size"]:
        logits = self.W_U(x)
        return logits

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.num_heads * cfg.d_head, cfg.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #type: ignore
        self.n_head = cfg.num_heads
        self.d_head = cfg.d_head

    #@jaxtyped(typechecker=beartype) TODO torch.compile doesn't work with jaxtyped decorators 
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_len d_model"],
        attn_mask: Optional[Bool[Tensor, "batch sequence_len sequence_len"]] = None,
    ) -> Float[Tensor, "batch sequence_len d_model"]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None if self.cfg.is_causal else attn_mask, 
            is_causal=self.cfg.is_causal
        )
        y = attn.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, cfg: BaseTransformerConfig):
        super().__init__()
        hidden_dim = cfg.d_model*cfg.d_mult

        self.w1 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden_dim, bias=False)
        self.act_fn = cfg.nonlin

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        RMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.variance_epsilon = eps

    def forward(self, x : Float[Tensor, "batch sequence_len d_model"]):
        input_dtype = x.dtype

        #we perform normalization in float32
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)
    
from typing import Protocol

class SwiGlUConfig(Protocol):
    d_model: int
    nonlin: Union[nn.Module, Callable[[Tensor], Tensor]] #NOTE llama3 uses SILU 
    multiple_of: int
    ffn_dim_multiplier: Optional[float]

#LLAMA3 ffn layer
class SwiGLU_MLP(nn.Module):
    def __init__(
        self,
        config : SwiGlUConfig
    ):
        super().__init__()
        self.nonlin = config.nonlin
        hidden_dim = int(2 * config.d_model / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=False)

    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"]
    ) -> Float[Tensor, "batch sequence_len d_model"]:
        return self.w2(self.nonlin(self.w1(x)) * self.w3(x))


#roatry embeddings from huggingface
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, :, : x.shape[-2], :]
        sin = sin[:, :, : x.shape[-2], :]

        return (x * cos) + (self.rotate_half(x) * sin)

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            self.apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            self.apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )

#from https://github.com/meta-llama/llama3/blob/bf8d18cd087a4a0b3f61075b7de0b86cf6c70697/llama/model.py#L90
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

ACT2CLS = {
    "gelu": (nn.GELU, {'approximate': 'tanh'}) ,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

ACT2FN = ClassInstantier(ACT2CLS)




class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : int,
        out_features : Optional[int] = None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y