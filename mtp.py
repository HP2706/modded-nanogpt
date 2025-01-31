#efficient implementation of multi-token prediction for training
#https://github.com/linkedin/Liger-Kernel/blob/main/examples/medusa/medusa_util.py#L87
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
""" lce = LigerFusedLinearCrossEntropyLoss()


for i in range(model.medusa_num_heads + 1):
    shift_hidden_states = (
        hidden_states[..., : -(1 + i), :].contiguous().view(-1, model.config.hidden_size)
    )
    shift_labels = labels[..., (1 + i) :].contiguous().view(-1)

    weight = model.lm_head.weight if i == 0 else model.medusa_head[i - 1][-1].weight
    loss_i = lce(weight, shift_hidden_states, shift_labels)

    loss += calculate_loss_contribution(
        loss_i,
        i,
        medusa_only_heads,
        medusa_decay_coefficient,
        medusa_heads_coefficient,
        medusa_scheduler_coefficient,
    ) """
    
import torch

class MTP(torch.nn.Module):
    def __init__(
        self, 
        n_tokens : int,
        d_hidden : int,
        d_vocab : int,
        use_liger : bool = True,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_hidden = d_hidden
        self.d_vocab = d_vocab
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(d_hidden, d_vocab) for _ in range(n_tokens)
        ])
        self.use_liger = use_liger
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
        
        for i in range(self.n_tokens):
            shift_labels = labels[..., (1 + i) :].contiguous().view(-1) # remove the first i tokens
            head : torch.nn.Linear = self.heads[i]
            if self.use_liger:
                loss = self.loss_fn.forward(
                    head.weight,
                    shared_trunk, # merge batch and sequence dimensions
                    shift_labels,
                    bias=head.bias,
                )
                
            else:
                logits = head.forward(shared_trunk)
                loss =self.loss_fn.forward(logits.view(-1, self.d_vocab), shift_labels)
            
            #NOTE we call backward multiple times to accumulate the gradients to d(the shared trunk)
            loss.backward()
            
        hidden_states.backward(gradient=shared_trunk.grad)
        