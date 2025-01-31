from models.mtp_model import DeepSeekV3MTP
import torch.nn as nn
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
torch.manual_seed(0)
# test if the gradients accumulate correctly

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

emb = nn.Linear(10, 8).to(device)
unembed = nn.Linear(10, 128).to(device)
input = torch.randn(1, 10, 10).to(device) # 1 batch size, 10 tokens, 10 features
labels = torch.randint(0, 128, (1, 10)).to(device)

n_tokens = 2 # two mtp tokens
mtp = DeepSeekV3MTP(
    n_tokens=n_tokens,
    d_hidden=8,
    d_vocab=128,
    num_heads=1,
    use_liger=False,
    proj_fp8=False,
).to(device)


from torch.nn.attention.flex_attention import create_block_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Because the sparsity pattern is independent of batch and heads, we'll set them to None (which broadcasts them) 
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=8, KV_LEN=8)
x = x0 = emb.forward(input)

hidden_states = x[:, :-n_tokens]

print('hidden_states', hidden_states.shape)
print('x', x.shape)
print('x0', x0.shape)
print('labels', labels.shape)
d = x.detach()
d.requires_grad = True
loss, shared_trunk_grad = mtp.forward1(
    d, 
    x0, 
    block_mask,
    labels
)

grad = d.backward(gradient=shared_trunk_grad)
x.backward(gradient=grad)

print("emb.weight.grad", emb.weight.grad.norm())

torch.zero_grad()
""" 
loss, _ = mtp.forward(
    hidden_states, 
    x0, 
    torch.ones(8, 8), 
    labels)

loss.backward()

print("emb.weight.grad", emb.weight.grad.norm()) """