import time
from types import SimpleNamespace
from pydantic import Field, model_validator, root_validator
from typing import Any, List, Literal, Optional, Union, cast
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from Models.Blocks import (
    MixtralBlockSparseTop2MLP,
    UnEmbedding,
    MultiHeadAttention,
    MLP
)
from Models.LLMS.VanillaTransformer import VanillaTransformerBlock
from Models.LLMS.LLMBase import ModelMixin, ModelOutputMixin, TransformerMixin
from Models.LLMS.configs import BaseTransformerConfig, ModelConfig
from jaxtyping import jaxtyped, Int, Float
from beartype import beartype

class MoEOutput(ModelOutputMixin):
    #the sum of these losses is the loss term in the model
    ce_loss: Optional[float] = None # "cross_entropy_loss"
    auxiliary_loss: Optional[float] = None #"auxiliary_loss for model router"
    router_z_loss: Optional[float] = None #"router_z_loss for model router"
    routing_scores : Optional[float] = None #"routing_scores for model router"
    routing_logits_norm : Optional[Tensor] = None 
    #"routing_logits_norm for model logits pre router
    #this should decrease incentivized by the z loss."

class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, cfg : "MoEConfig"):
        super().__init__()
        self.hidden_dim = cfg.d_model
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(cfg) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = cfg.jitter_noise

    def compute_router_z_loss( #TODO implement this in forward pass
        self,
        pre_router_logits : Float[Tensor, "batch_x_sequence_len num_experts"],
    ) -> Float[Tensor, "1"]:
        #loss as explained in
        #https://cameronrwolfe.substack.com/p/conditional-computation-the-birth
        #we use this approach to avoid inf or nan values.
        max_logits = torch.max(pre_router_logits, dim=-1, keepdim=True).values
        exp_logits = torch.exp(pre_router_logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=-1)
        log_sum_exp = torch.log(sum_exp) + max_logits.squeeze(-1)
        
        # Compute the squared term
        squared_term = log_sum_exp ** 2
        router_z_loss = self.betas * torch.mean(squared_term + self.cfg.eps)
        return router_z_loss

    #inspired by 
    # https://github.com/huggingface/transformers/blob/1218e439b5fe05423693407653a1fb064263fef4/src/transformers/models/mixtral/modeling_mixtral.py#L870
    def forward(
        self, 
        x: Float[Tensor, "batch sequence_len d_model"]
    ) -> tuple[
            Float[Tensor, "batch sequence_len d_model"], 
            Float[Tensor, "batch_x_sequence_len num_experts"],
    ]:
        B, C, D = x.shape
        if self.training and self.jitter_noise:
            x *= torch.empty_like(x).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        
        x = x.view(-1, D)
        # router_logits: (batch * C, n_experts)
        router_logits = self.gate(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros(
            (B * C, D), dtype=x.dtype, device=x.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, D)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(B, C, D)

        return final_hidden_states, router_logits

class MoETransformerBlock(nn.Module):
    def __init__(
        self, 
        cfg: "MoEConfig", 
    ):
        super().__init__()
        self.cfg = cfg
        self.moe_mlp = MixtralSparseMoeBlock(cfg)
        self.attn = MultiHeadAttention(cfg)
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.ln_2 = nn.LayerNorm(cfg.d_model)

    def forward(
        self, 
        x : Float[Tensor, "batch sequence_len d_model"]
    ) -> tuple[
        Float[Tensor, "batch sequence_len d_model"], 
        Float[Tensor, "batch_x_sequence_len num_experts"],
    ]:
        x = x + self.attn.forward(self.ln_1(x))
        residual, router_gate_logits = self.moe_mlp.forward(self.ln_2(x))
        x = x + residual
        return x, router_gate_logits

class MoETransformer(TransformerMixin):
    def __init__(self, cfg: "MoEConfig", is_master_process: bool = True):
        super().__init__(cfg, is_master_process)
        self.cfg = cfg
        self.model = nn.ModuleDict(dict(
            embedding = self.embedding,
            pos_embed = self.pos_embed,
            layers = nn.ModuleList([MoETransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.model.embedding.weight = self.lm_head.weight

        #pydantic validators ensure either beta or betas is set
        betas = [self.cfg.beta] * self.cfg.n_layers if self.cfg.beta is not None else self.cfg.betas
        self.betas = cast(Tensor, Tensor(betas))

        #pydantic validators ensure either alpha or alphas is set
        alphas = [self.cfg.alpha] * self.cfg.n_layers if self.cfg.alpha is not None else self.cfg.alphas
        self.alphas = cast(Tensor, Tensor(alphas))
        self.apply(self._init_weights)

    def compute_aux_loss(
        self, 
        router_logits : Float[Tensor, "n_layers batch_x_sequence_len num_experts"],
        expert_mask : Optional[Float[Tensor, "n_layers num_experts topk batch_x_sequence_len "]] = None,
    ) -> Float[Tensor, "1"]:
        '''computes the auxiliary loss over all layers at once'''
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        
        if expert_mask:
            assert expert_mask.shape[-1] == router_logits.shape[1], (
                f"expert_mask and router_logits must have same, "
                f"token count got {expert_mask.shape[-1]} and {router_logits.shape[1]}" 
            ) 
        else:
            _, selected_experts = torch.topk(routing_weights, self.cfg.top_k, dim=-1)
            expert_mask = torch.nn.functional.one_hot(selected_experts, self.cfg.num_experts)
        
        expert_mask = expert_mask.view(self.cfg.n_layers, -1, self.cfg.top_k, self.cfg.num_experts).float()
        # we normalize by top_k to get probabilities
        tokens_per_expert = (torch.sum(expert_mask, dim=2) / self.cfg.top_k).mean(0)
        
        # Compute the average probability of routing to these experts
        self.alphas = self.alphas.to(tokens_per_expert.device) 
        scaled_routing_weights = routing_weights * self.alphas.unsqueeze(-1).unsqueeze(-1)
        router_prob_per_expert = torch.mean(scaled_routing_weights, dim=0)
        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
        return overall_loss # * self.cfg.num_experts #TODO should this be here??
    
    def forward(
        self, 
        idx : Int[Tensor, "B T"], 
        targets : Optional[Int[Tensor, "B T"]] = None
    ) -> MoEOutput:
        self.check_forward(idx, targets)
        B, T = idx.shape
        assert T <= self.cfg.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.cfg.n_ctx}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos_emb = self.model.pos_embed(pos) 
        tok_emb = self.model.embedding(idx) 
        x : Tensor = tok_emb + pos_emb

        layer_router_gate_logits = []

        for layer_num, layer in enumerate(cast(List[MoETransformerBlock], self.model.layers)):
            if self.cfg.is_training or targets is not None:
                x, router_gate_logits = layer.forward(x)
                layer_router_gate_logits.append(router_gate_logits) #type: ignore
            else:
                x, _ = layer.forward(x)
        x = self.model.ln_f(x)
        x = self.lm_head(x)

        if targets is None:
            return MoEOutput(logits=x)
        else:
            router_gate_logits = torch.stack(layer_router_gate_logits, dim=0)
            cross_entropy_loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
            auxiliary_loss = self.compute_aux_loss(router_gate_logits, expert_mask=None) #type: ignore
            total_loss = cross_entropy_loss + auxiliary_loss  #+ router_z_loss_accum TODO add optionally
            routing_logits_norm = torch.norm(router_gate_logits, dim=-1).mean(0).cpu()

            return MoEOutput(
                logits=x.view(B, T, -1),
                loss=total_loss,
                ce_loss=cross_entropy_loss,
                auxiliary_loss=auxiliary_loss.detach(),
                #router_z_loss=router_z_loss_accum,
                routing_logits_norm=routing_logits_norm,
            )


class MoEConfig(BaseTransformerConfig):
    num_experts: int
    top_k: int
    jitter_noise: Optional[float] = None
    alpha: Optional[float] = Field(
        default=None,
        description="this is the aux loss weight",
    )
    alphas: Optional[list[float]] = Field(
        default=None,
        description=(
            "this is the aux loss weight but where a different weight can be applied to each layer"
        ),
    )

    # beta/betas inspired by router z loss coeff in paper: https://arxiv.org/pdf/2202.08906
    beta: Optional[float] = Field(
        default=None,
        description=(
            "the router z loss; if beta is chosen the same loss is applied to each layer"
        ),
    )
    betas: Optional[list[float]] = Field(
        default=None,
        description=(
            "the router z loss; if betas is chosen a different loss is applied to each layer"
        ),
    )
    gate_type: Literal["mixtral"] = "mixtral"
    gate_norm_factor: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def validate_alphas_betas(cls, values):
        n_layers = values.get("n_layers")
        if values.get("alphas") is not None and values.get("betas") is not None:
            assert len(values.get("alphas")) == n_layers and len(values.get("betas")) == n_layers

        if not values.get("alpha") and not values.get("alphas"):
            raise ValueError("Either alpha or alphas must be set.")
        if not values.get("beta") and not values.get("betas"):
            raise ValueError("Either beta or betas must be set.")

        # disambiguate
        if values.get("alpha") and values.get("alphas"):
            raise ValueError("Only one of alpha or alphas can be set, not both.")
        if values.get("beta") and values.get("betas"):
            raise ValueError("Only one of beta or betas can be set, not both.")
        return values
