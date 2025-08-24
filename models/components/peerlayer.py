import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float
from einops import einsum

class PeerLayer(nn.Module):
    def __init__(
        self, 
        d_in : int,
        d_out : int,
        d_query : int,
        query_heads : int,
        n_experts : int,
        topk : int = 10,
    ):
        super().__init__()
        assert d_query % 2 == 0, "d_query must be even"
        N_sqrt = int(torch.sqrt(torch.tensor(n_experts)))
        assert N_sqrt * N_sqrt == n_experts, "n_experts must be a perfect square"
        self.n_sqrt = N_sqrt
        self.K1 = nn.Parameter(torch.randn(N_sqrt, d_query // 2)) 
        self.K2 = nn.Parameter(torch.randn(N_sqrt, d_query // 2)) 
        self.n_heads = query_heads
        self.d_query_half = d_query // 2
        self.query_network = nn.Linear(d_in, d_query * query_heads)
        self.topk = topk
        self.W_up = nn.Embedding(n_experts, d_out)
        self.W_down = nn.Embedding(n_experts, d_in)
        
    def cartesian_product(
        self, 
        topk_q1_k1 : Float[Tensor, "batch seq n_heads topk"], 
        topk_q2_k2 : Float[Tensor, "batch seq n_heads topk"],
    ) -> Float[Tensor, "batch seq topk^2"]:
        i_values = topk_q1_k1.unsqueeze(-1).expand(-1, -1, self.n_heads, -1, self.topk)  # [batch, seq, topk, topk]
        j_values = topk_q2_k2.unsqueeze(-2).expand(-1, -1,self.n_heads, self.topk, -1)  # [batch, seq, topk, topk]
        vals = i_values * self.n_sqrt + j_values  # [batch, seq, topk, topk]
        kandidate_keys = vals.reshape(topk_q1_k1.shape[:-1] + (self.topk**2,)) # [batch, seq, topk^2]
        return kandidate_keys
        
    def get_indices(
        self, 
        queries : Float[Tensor, "batch seq n_heads d_query"]
    )-> tuple[
        Float[Tensor, "batch seq n_heads topk"], 
        Float[Tensor, "batch seq n_heads topk"]
    ]:
        q1, q2 = torch.split(queries, self.d_query_half, dim=-1) # [batch, seq, n_heads, query_dim // 2]
        
        q1_k1 = torch.matmul(q1, self.K1.T) # [batch, seq, n_heads, N_sqrt]
        q2_k2 = torch.matmul(q2, self.K2.T) # [batch, seq, n_heads, N_sqrt]

        topk_q1_k1, topk_indices_q1_k1 = torch.topk(q1_k1, k=self.topk, dim=-1) 
        topk_q2_k2, topk_indices_q2_k2 = torch.topk(q2_k2, k=self.topk, dim=-1)
        
        # Calculate all scores and indices using cartesian product
        batch_size, seq_len = queries.shape[0], queries.shape[1]
        
        # Expand for cartesian product
        all_scores = (
            topk_q1_k1.unsqueeze(-1).expand(-1, -1, self.n_heads, -1, self.topk) +
            topk_q2_k2.unsqueeze(-2).expand(-1, -1, self.n_heads, self.topk, -1)
        ).view(batch_size, seq_len, self.n_heads, -1)  # [batch, seq, n_heads, topk^2]
        
        all_indices = (
            topk_indices_q1_k1.unsqueeze(-1).expand(-1, -1, self.n_heads, -1, self.topk) * self.n_sqrt +
            topk_indices_q2_k2.unsqueeze(-2).expand(-1, -1, self.n_heads, self.topk, -1)
        ).view(batch_size, seq_len, self.n_heads, -1)  # [batch, seq, n_heads, topk^2]
        
        # Select top-k from the cartesian product
        vals, best_indices = torch.topk(all_scores, k=self.topk, dim=-1)  # [batch, seq, n_heads, topk]
        expert_indices = all_indices.gather(-1, best_indices)  # [batch, seq, n_heads, topk]
        
        return vals, expert_indices

    def forward(self, x : Float[Tensor, "batch seq d_in"]):

        queries = self.query_network(x) # [batch, seq, n_heads, query_dim]
        values, indices = self.get_indices(queries.view(queries.shape[0], queries.shape[1], self.n_heads, -1))

        w_down = self.W_down(indices)
        w_up = self.W_up(indices)
        scores = torch.softmax(values, dim=-1)
        x = einsum(x , w_down, 'b seq d_in, b seq n_heads topk d_in -> b seq n_heads topk' )
        x = F.gelu(x)
        x = x * scores
        x = einsum(x , w_up, 'b seq n_heads topk, b seq n_heads topk d_out -> b seq d_out' )
        return x
        