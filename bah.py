#show multitoken gradient accumulation works
from copy import deepcopy
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
def norm(x):
    result = F.rms_norm(x, (x.size(-1),))
    return result


class DeepSeekV3MTP(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        d_hidden: int,
        d_vocab: int,
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
            nn.Linear(d_hidden, d_hidden)
            for _ in range(n_tokens)
        ])
        # Projection layer combining two normalized inputs
        self.projs = nn.ModuleList([
            nn.Linear(d_hidden * 2, d_hidden)
            for _ in range(n_tokens)
        ])
        
        # Output head (shared)
        self.shared_head = nn.Linear(d_hidden, d_vocab)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(
        self, 
        shared_trunk: Tensor, 
        x0: Tensor, 
        block_mask: Tensor, 
        labels: Tensor,
        with_backward: bool = True
    ):
        """
        shared_trunk: (B, T, H)
        x0: (B, T+N, H) -- contains the token-specific embeddings for each prediction step
        labels: (B, T+N)
        """
        T = shared_trunk.shape[1]
        assert shared_trunk.shape[1] == labels.shape[1] - self.n_tokens
        assert x0.shape[1] - self.n_tokens == T

        if not with_backward:
            # Regular forward: compute token predictions sequentially with the full graph.
            current_hidden = shared_trunk
            loss_accum = 0.0
            for i in range(self.n_tokens):
                embs = x0[..., i:i+T, :]
                shift_labels = labels[..., i:i+T].contiguous().view(-1)
                x_norm = norm(current_hidden)
                next_norm = norm(embs)
                combined = torch.cat([x_norm, next_norm], dim=-1)
                h_prime = self.projs[i](combined)
                current_hidden = self.blocks[i](h_prime)
                logits = self.shared_head(current_hidden)
                loss = self.loss_fn(logits.view(-1, self.d_vocab), shift_labels)
                loss_accum += loss
            return loss_accum

        else:
            # Revised backward: we do a token-by-token backward pass
            total_loss = 0.0
            # Start with the shared trunk as the root.
            current_hidden = shared_trunk
            # Initialize accumulated_grad with zeros instead of None.
            accumulated_grad = torch.zeros_like(current_hidden)
            for i in range(self.n_tokens):
                embs = x0[..., i:i+T, :]
                shift_labels = labels[..., i:i+T].contiguous().view(-1)

                # Ensure current_hidden tracks gradient.
                current_hidden.requires_grad_(True)
                x_norm = norm(current_hidden)
                next_norm = norm(embs)
                combined = torch.cat([x_norm, next_norm], dim=-1)
                h_prime = self.projs[i](combined)
                h_next = self.blocks[i](h_prime)
                logits = self.shared_head(h_next)
                loss = self.loss_fn(logits.view(-1, self.d_vocab), shift_labels)
                total_loss = total_loss + loss

                # Call backward on the loss so that parameter gradients are accumulated.
                loss.backward(retain_graph=True)
                
                # Compute the gradient wrt current_hidden (local contribution)
                local_grad = torch.autograd.grad(loss, current_hidden, retain_graph=True)[0]
                # Compute the extra gradient from the fact that h_next depends on current_hidden:
                extra = torch.autograd.grad(
                    h_next, current_hidden, grad_outputs=accumulated_grad, retain_graph=True
                )[0]
                grad_total = local_grad + extra

                # Prepare for the next iteration.
                accumulated_grad = grad_total.detach()
                # Detach h_next to free the computation history, but keep its connection to parameters already updated.
                current_hidden = h_next.detach()

            # Finally, propagate the accumulated gradient back into the original shared trunk.
            shared_trunk.backward(accumulated_grad)
            return total_loss
    
def test_mtp():
    torch.manual_seed(42)
    
    # Model and data parameters
    hidden_dim = 20
    num_heads = 2
    num_classes_per_head = 4
    batch_size = 5
    seq_len = 8  # Base sequence length
    use_liger = False

    model = DeepSeekV3MTP(num_heads, hidden_dim, num_classes_per_head, use_liger=use_liger, proj_fp8=False)
    # Create input tensor with requires_grad=True
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    # Create random target classes with extended sequence length
    y = torch.randint(0, num_classes_per_head, (batch_size, seq_len + num_heads))
    
    # Test regular forward
    x0 = torch.randn(batch_size, seq_len+num_heads, hidden_dim)
    block_mask = torch.ones(batch_size, seq_len)
    loss = model.forward(
        shared_trunk=x,
        x0=x0,
        block_mask=block_mask,
        labels=y,
        with_backward=False
    )
    print(loss.item())
    loss.backward()
    
    
    param_grads = [(name, param.grad.clone().detach()) for name, param in model.named_parameters()]

    # Test forward_backward
    print("param_grads", [(name, param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None])
    model.zero_grad()
    
    x.requires_grad = True
    loss = model.forward(
        shared_trunk=x,
        x0=x0,
        block_mask=block_mask,
        labels=y,
        with_backward=True
    )
    print("loss2", loss.item())
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad  # Loss should have grad enabled
    print("param_grads manual bwd", [(name, param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None])
    
    named_params = list(model.named_parameters())
    for i in range(len(named_params)):
        is_correct = torch.allclose(named_params[i][1].grad, param_grads[i][1])
        if not is_correct:
            print("Gradient mismatch for", named_params[i][0], "got", named_params[i][1].grad.norm(), "expected", param_grads[i][1].norm())
        else:
            print("Gradient match for", named_params[i][0])
            
    #all_same = all(torch.allclose(param.grad, params[i].grad) for i, param in enumerate(model.parameters()))
    #assert all_same, f"Gradients are not the same for all parameters"
    # Verify gradients were computed
    print(f"MTP test passed with use_liger={use_liger}")
    print(f"Loss value: {loss.item()}")

if __name__ == "__main__":
    test_mtp()