# the full nanogpt file 
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass, asdict
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention
import wandb
from torchao.optim import AdamW8bit

class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        # Project from model dimension to per-head concatenated dimension
        self.Q_w = nn.Linear(dim, hdim, bias=False)
        self.K_w = nn.Linear(dim, hdim, bias=False)
        self.V_w = nn.Linear(dim, hdim, bias=False)
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = nn.Linear(hdim, dim, bias=True)
        self.attn_scale = 0.12

    def forward(self, x: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        
        q = self.Q_w(x).view(B, T, self.num_heads, self.head_dim)
        k = self.K_w(x).view(B, T, self.num_heads, self.head_dim)
        v = self.V_w(x).view(B, T, self.num_heads, self.head_dim)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = nn.Linear(dim, hdim)
        self.c_proj = nn.Linear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, block_mask: BlockMask):
        # Some layers intentionally skip attention (self.attn can be None)
        if self.attn is not None:
            x = x + self.attn(norm(x), block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight.detach().zero_() # 
        assert num_layers % 2 == 0
        pad = (-num_layers * 5) % dist.get_world_size()
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers), # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)], # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)], # SA lambdas
            torch.ones(pad),
        ]))
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        
        def build_bm() -> BlockMask:
            return BlockMask.from_kv_blocks(
                partial_kv_num_blocks,
                partial_kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
            
        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm()

    def forward(self, input_seq: Tensor, target_seq: Tensor):
        assert input_seq.ndim == 1

        bm = self.create_blockmasks(input_seq)
        x = norm(self.embed(input_seq)[None])

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, bm)
        x = norm(x)

        logits = self.lm_head(x).float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

# find world_size starting indicies, such that each begins with token 50256 and local_batches don't overlap
def find_batch_starts(tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= local_batch_size:
            starts.append(start) # append start once end pos is confirmed
            if len(starts) == dist.get_world_size():
                return starts, end - pos
            start = end
    assert False # increase max_batch_span if necessary

def distributed_data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size # provide buffer to handle samples up to length local_batch_size
    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(tokens, pos, local_batch_size, max_batch_span)
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size
        buf = tokens[start_idx:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_span
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main
# number of local GPUs and gradient accumulation to emulate 8-GPU global batch
ngpus = torch.cuda.device_count()
target_ngpus = 8
# accumulate enough micro-steps to match the effective global batch of 8 GPUs
from math import ceil

import datetime
@dataclass
class Hyperparameters:
    # data
    train_files = "/root/data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "/root/data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 24*1024 # FlexAttention sequence length
    val_seq_len = 2*64*1024 # FlexAttention sequence length for validation
    # optimization
    num_iterations = 3500 # number of iterations to run
    cooldown_frac = 0.45 # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every = 250 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False
    logs_path = f'/root/sandbox/logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # logging granularity (match optimized patterns)
    log_every_tokens: int = 50_000
    
args = Hyperparameters()
os.makedirs(args.logs_path, exist_ok=True)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])  # number of distributed processes

assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs(args.logs_path, exist_ok=True)
    logfile = f"{args.logs_path}/{run_id}.txt"
    print(logfile)
    
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

# Initialize Weights & Biases on master process
if master_process:
    _run_name = f"unoptimized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="modded-nanogpt",
        name=_run_name,
        config=asdict(args),
        reinit=True,
    )

model: nn.Module = GPT(
    vocab_size=50257, 
    num_layers=12, 
    num_heads=6, 
    model_dim=768, 
    max_seq_len=max(args.train_seq_len, args.val_seq_len)
).cuda().bfloat16()

for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# number of local GPUs and gradient accumulation to emulate 8-GPU global batch
ngpus = torch.cuda.device_count()
target_ngpus = 8
from math import ceil
grad_accum_steps = max(1, ceil(target_ngpus / max(1, ngpus)))
assert world_size == ngpus, "Expected one process per GPU on a single node"

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer = DistAdam(model.parameters(), lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1


#model: nn.Module = torch.compile(model, dynamic=False)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(
    model=copy.deepcopy(model.state_dict()),
    optimizer=copy.deepcopy(optimizer.state_dict())
) # save the initial state

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    # single warmup step to compile kernels
    model(inputs, targets).backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    
model.load_state_dict(initial_state["model"])
optimizer.load_state_dict(initial_state["optimizer"])
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
training_time_ms = 0
# token accounting
tokens_seen = 0
next_tokens_log = args.log_every_tokens
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        # mirror to wandb
        if master_process:
            wandb.log({
                "val/loss": float(val_loss.item()),
                "time/train_ms": float(training_time_ms),
                "time/step_avg_ms": float(training_time_ms / max(step, 1)),
                "step": step,
            }, step=step)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, 
                       code=code, 
                       model=model.state_dict(), 
                       optimizer=optimizer.state_dict())
            os.makedirs(f"{args.logs_path}/{run_id}", exist_ok=True)
            torch.save(log, f"{args.logs_path}/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    # Accumulate gradients over multiple micro-steps to emulate 8-GPU global batch
    train_log = False
    for micro in range(grad_accum_steps):
        inputs, targets = next(train_loader)
        loss = model(inputs, targets)
        loss = loss / grad_accum_steps
        loss.backward()
        # token accounting per micro-step
        tokens_seen += (world_size * args.train_seq_len)
        if master_process and tokens_seen >= next_tokens_log:
            train_log = True

    optimizer.step()
    # set optimization hyperparameters
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    if train_log:
        print0(f"tokens_seen milestone: {tokens_seen}")
        if master_process:
            wandb.log({
                "train/tokens_seen": int(tokens_seen),
                "train/step": int(step),
                "train/micro_step": int(micro + 1),
            }, step=step)
        # schedule next milestone
        k = max(1, tokens_seen // args.log_every_tokens)
        next_tokens_log = k * args.log_every_tokens + args.log_every_tokens

    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)
    # mirror to wandb
    if master_process:
        current_lr = None
        if len(optimizer.param_groups) > 0:
            current_lr = float(optimizer.param_groups[0].get("lr", 0.0))
        wandb.log({
            "lr/adam": current_lr,
            "time/train_ms": float(approx_training_time_ms),
            "time/step_avg_ms": float(approx_training_time_ms / (step + 1)),
            "step": step + 1,
        }, step=step + 1)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
if master_process:
    wandb.log({
        "cuda/max_memory_allocated_mib": float(torch.cuda.max_memory_allocated() // 1024 // 1024),
        "cuda/max_memory_reserved_mib": float(torch.cuda.max_memory_reserved() // 1024 // 1024),
    })
dist.destroy_process_group()
