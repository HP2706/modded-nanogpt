#an implementation of the ultra sparse memory transformer
#paper: https://arxiv.org/pdf/2411.12364
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, cast
from jaxtyping import Float, Int
from einops import einsum


def binary_topm(
    x: Float[Tensor, "... n"],  # Input tensor with arbitrary leading dimensions and n elements
    m: int,                     # Number of top elements to select
) -> tuple[Float[Tensor, "... n"], Int[Tensor, "... n"]]:    # Binary tensor with same shape as input
    """
    Convert top-m elements to 1 and others to 0 along the last dimension.
    
    Args:
        x: Input tensor
        m: Number of top elements to select
    
    Returns:
        Binary tensor where top-m elements are 1 and others are 0
    """
    # Get indices of top-m elements
    _, topk_indices = torch.topk(x, m, dim=-1)
    
    # Create zero tensor with same shape as input
    binary = torch.zeros_like(x)
    
    # Set top-m elements to 1
    binary.scatter_(-1, topk_indices, 1.0)
    
    return binary, topk_indices

class UltraSparseMemoryMLP(nn.Module):
    def __init__(
        self,
        d_value: int,
        d_query: int,
        r : int,
        query_heads: int,
        n_experts: int,
        num_cores: int = 4,  # H (head number) in the paper
        topk: int = 10,      # m (activated value number) in the paper
        virtual_expansion_factor: int = 4,  # E in the paper
        num_layers: int = 1,  # L in the paper
    ):
        super().__init__()
        assert math.sqrt(virtual_expansion_factor).is_integer(), "virtual_expansion_factor must be a perfect square"
        self.d_value = d_value
        self.d_query = d_query
        self.query_heads = query_heads
        self.n_experts = n_experts
        #self.num_cores = num_cores
        self.topk = topk
        self.r = r
        
        self.virtual_expansion_factor = virtual_expansion_factor
        self.virtual_expansion_factor_sqrt = int(math.sqrt(virtual_expansion_factor)) # type: ignore
        self.d_k = d_query*self.virtual_expansion_factor_sqrt 
        
        self.Cs = nn.ParameterList([nn.Parameter(torch.randn(r, r)) for _ in range(num_cores)])
        
        self.C = torch.stack([p for p in self.Cs]).sum(dim=0) # we just sum the cores
        self.K_row = nn.Parameter(torch.randn(r, n_experts, self.virtual_expansion_factor_sqrt, self.d_query // (r * 2)))
        self.K_col = nn.Parameter(torch.randn(r, n_experts, self.virtual_expansion_factor_sqrt, self.d_query // (r * 2)))
        # these are input dependent 
        
        # Get leading singular vectors (first column of U and V)
        # Reshape to [r, 1] for matrix operations
        
        # Initialize with proper scaling factor
        # N(0, E/(2mHL)) where:
        # E = virtual_expansion_factor
        # m = topk
        # H = num_cores
        # L = num_layers
        scale = (virtual_expansion_factor / (2 * topk * r * num_layers)) ** 0.5
        
        # Split physical memory into cores V = [V^(1), ..., V^(h)]
        d_value_per_core = d_value // num_cores
        self.physical_memories = nn.ParameterList([
            nn.Parameter(torch.randn(n_experts, d_value_per_core) * scale)
            for _ in range(num_cores)
        ])
        
        # Virtual projectors for each core and expansion
        self.virtual_projectors = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.randn(d_value_per_core, d_value_per_core) * scale)
                for _ in range(virtual_expansion_factor)
            ])
            for _ in range(num_cores)
        ])
        
    @property
    def T(self):
        _, _, T = torch.svd(self.C)
        return T[:1]
    
    @property
    def U(self):
        U, _, _ = torch.svd(self.C)
        return U[:1]
        
    def improved_tucker_decomp(
        self, 
        Q_row: Float[Tensor, "r exp d_k_r"],
        Q_col: Float[Tensor, "r exp d_k_r"],
        m : int,
    ) -> tuple[Float[Tensor, "n m"], Int[Tensor, "n m"]]:
        """
        Multi-core scoring version of TDQKR
        """
        assert self.d_query % self.r == 0
        
        # these are input dependent 
        # NOTE we approximate C as U @ T.T EQ 9        


        S_col = einsum(self.K_col, Q_col, "r n exp dk_div_r, bs r exp dk_div_r -> bs r exp n") # [r, n, d_k // r] X [..., r, d_k // r] -> [..., r, n]
        S_row = einsum(self.K_row, Q_row, "r n exp dk_div_r, bs r exp dk_div_r -> bs r exp n") # [r, n, d_k // r] X [..., r, d_k // r] -> [..., r, n]

        
        # T.T @ S_col shape: [1, r] @ [bs, r, exp, n] -> [bs, exp, n]
        scores_col = einsum(self.T.squeeze(0), S_col, "r, bs r exp n -> bs exp n")  # [bs, exp, n]
        mask_col, topk_indices_col = binary_topm(scores_col, m)  # [bs, exp, n]
        
        # Reshape mask_col to match S_col's dimensions
        mask_col = mask_col.unsqueeze(1).expand(-1, self.r, -1, -1)  # [bs, r, exp, n]
        S_approx_col = mask_col * S_col  # Element-wise multiplication

        # Similarly for row
        scores_row = einsum(self.U.squeeze(0), S_row, "r, bs r exp n -> bs exp n")  # [bs, exp, n]
        mask_row, topk_indices_row = binary_topm(scores_row, m)  # [bs, exp, n]
        
        # Reshape mask_row to match S_row's dimensions
        mask_row = mask_row.unsqueeze(1).expand(-1, self.r, -1, -1)  # [bs, r, exp, n]
        S_approx_row = mask_row * S_row  # Element-wise multiplication

        # Compute individual score maps for each core
        all_scores = []
        for C_i in self.Cs:
            grid_i = einsum(
                S_approx_row, 
                C_i, 
                S_approx_col, 
                "bs r exp_i n, r r, bs r exp_j n -> bs exp_i exp_j n"
            )
            grid_i = grid_i.view(grid_i.shape[0], self.virtual_expansion_factor, grid_i.shape[-1])
            all_scores.append(grid_i)
        
        # Sum scores across cores (Equation 16)
        grid = sum(all_scores)
        Scores, Indices = torch.topk(torch.softmax(grid, dim=-1), m, dim=-1)
        return Scores, Indices

    def access_virtual_memory(
        self,
        values: Float[Tensor, "b_x_seq_len exp m"],
        indices: Int[Tensor, "b_x_seq_len exp m"],
    ) -> Float[Tensor, "b_x_seq_len d_out"]:
        """
        Access the virtual memory with multi-core scoring
        """
        permuted_indices = torch.randperm(indices.shape[-1])
        values = values[..., permuted_indices]
        indices = indices[..., permuted_indices]
        
        out: Optional[Tensor] = None
        
        # Process each core separately
        for core_idx, physical_memory in enumerate(self.physical_memories):
            core_out: Optional[Tensor] = None
            
            for p in range(self.virtual_expansion_factor):
                gathered = physical_memory[indices[:, p, :]]
                vals = values[:, p, :]
                
                Non_virtual = einsum(gathered, vals, "bs m d_value, bs m -> bs d_value")
                if core_out is None:
                    core_out = Non_virtual @ self.virtual_projectors[core_idx][p]
                else:
                    core_out += Non_virtual @ self.virtual_projectors[core_idx][p]
            
            if out is None:
                out = core_out
            else:
                out = torch.cat([out, core_out], dim=-1)  # Concatenate along value dimension
        return cast(Tensor, out)
        
        
    def forward(
        self,
        Q: Float[Tensor, "b seq_len expansion_factor d_query"],
    ) -> Float[Tensor, "b seq_len d_out"]:
        
        # Split query into row and column components
        # Each should be [b, seq_len, d_query//2]
        mid = self.d_query // 2
        Q_row = Q[..., :mid]
        Q_col = Q[..., mid:]
        
        # Reshape for tucker decomposition
        # Each should become [b*seq_len, r, virtual_expansionm, d_query/(2*r)]
        Q_row = Q_row.reshape(-1, self.r, self.virtual_expansion_factor_sqrt, mid // self.r)
        Q_col = Q_col.reshape(-1, self.r, self.virtual_expansion_factor_sqrt, mid // self.r)
        
        vals, scores = self.improved_tucker_decomp(Q_row, Q_col, self.topk)
        
        out = self.access_virtual_memory(vals, scores)
       
        return out
        
    def aux_loss(self) -> Tensor:
        _, S, _ = torch.svd(self.C)
        # Get all singular values except the first one (i=2 to r)
        non_leading_singvals = S[1:]
        
        # Hyperparameters from the paper
        tau = 0.1  # threshold τ
        alpha = 0.1  # scaling factor α
        
        
        # Calculate max(0, λᵢ - τ)² for each non-leading singular value
        # and sum them up according to equation 12
        loss = (alpha / (len(S) - 1)) * torch.sum(torch.clamp(non_leading_singvals - tau, min=0) ** 2)
        return loss
        


if __name__ == "__main__":
    
    r = 4
    d_query = 16
    d_r = d_query // r
    model = UltraSparseMemoryMLP(
        d_value=16, 
        d_query=d_query, 
        r=r,
        query_heads=1, 
        n_experts=32, 
        topk=10, 
        virtual_expansion_factor=4
    )
    
    #query = torch.randn(10, 1, 16, 16)
    #print(model(query).shape)
    
    q = torch.randn(2, 10, 2, 16)
    print(model.forward(q))