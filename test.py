from train_gpt import GPT, Hyperparameters, Muon
import torch
args = Hyperparameters()

rank = 1
world_size = 1
model = GPT(vocab_size=2, num_layers=1, num_heads=1, model_dim=2)

hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]

optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)

if not args.use_adam_mini:
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]
    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizers = [optimizer1, optimizer2]
else:
    from adam_mini import Adam_mini
    embed_params = [(n,p) for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [(n,p) for n, p in model.named_parameters() if p.ndim < 2]
    head_params = [('lm_head.weight', model.lm_head.weight)]
    optimizers = [Adam_mini(
        dict(embed_params, lr=0.008),
        dict(scalar_params, lr=0.04),
        dict(head_params, lr=0.008),
        weight_decay=0.01
    )]




