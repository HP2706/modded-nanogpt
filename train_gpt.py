from operator import is_
from aiohttp import TraceConfig
import fire
import wandb
import datetime
from tarfile import BLOCKSIZE
from sympy import use
from tqdm import tqdm
from itertools import chain
import os
import math
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
from typing import Optional, Union
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
device = "cuda"
torch.empty(1, device=device, requires_grad=True).backward() # prevents a bug on some systems

import torch.distributed as dist
from torch import Tensor, nn
from models.Current_best_gpt import GPT
from utils import next_multiple_of_n, unwrap
from models.mtp_model import MTPGPT, MTP, DeepSeekV3MTP

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        assert all(isinstance(p, Tensor) for p in params)
        sizes = {p.numel() for p in params}
        def create_update_buffer(size: int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device=device)
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        param_groups = [
            dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_buffer = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(
                        g_world.view_as(p_world),
                        alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(
    filename_pattern: str, 
    batch_size: int, 
    rank : int, 
    world_size : int,
    from_path : Optional[str] = None
):
    print0(f"Starting data generator with batch_size={batch_size}, rank={rank}, world_size={world_size}")
    if from_path:
        base_path = Path(from_path)
        files = sorted(base_path.glob(filename_pattern))
    else:
        files = sorted(Path.cwd().glob(filename_pattern))
        
    print0(f"Found {len(files)} files matching pattern: {filename_pattern}", console=True)
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    print0("Loading first data shard...")
    tokens, pos = _load_data_shard(next(file_iter)), 0
    print0(f"Loaded initial shard with {len(tokens)} tokens")
    while True:
        if pos + batch_size + 1 >= len(tokens):
            print0("Loading next data shard...")
            tokens, pos = _load_data_shard(next(file_iter)), 0
            print0(f"Loaded new shard with {len(tokens)} tokens")
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------    
# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device(device, int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)

# check if A10g device
device_name = torch.cuda.get_device_name(device)
is_a10g = device_name == "NVIDIA A10"

dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()


print(f"Rank: {rank}, World size: {world_size}")
master_process = (rank == 0) # this process will do logging, checkpointing etc.

@dataclass
class TrainConfig:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"  # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"  # input .bin to eval validation loss on
    # optimization
    num_iterations: int = 1393  # number of iterations to run
    cooldown_frac: float = 0.4  # fraction of training spent cooling down the learning rate
    # evaluation and logging
    val_loss_every: int = 125  # every how many steps to evaluate val loss? 0 for only at the end
    # implementation
    seq_len: int = 64*(1024) if not is_a10g else 16*(1024)
    save_checkpoint: bool = False
    use_liger: bool = True
    use_adam_mini: bool = False
    use_mtp: bool = False
    num_layers: int = 12
    n_mtp_tokens: int = 2
    use_wandb: bool = True
    torch_compile: bool = True
    use_deepseek_mtp: bool = True
    proj_fp8: bool = False
    bfloat16: bool = False
    BLOCK_SIZE: int = 128
    IS_MODAL: bool = True

    def __post_init__(self):
        # Set values that depend on is_a10g
        self.num_heads = 6 if not is_a10g else 1  # NOTE there will be bugs if this is not 1 when model_dim != 768
        self.model_dim = 768 if not is_a10g else 32
        if self.use_mtp:
            self.seq_len = self.seq_len + self.n_mtp_tokens  # we add the n_tokens to the seq_len for the MTP
        
        self.batch_size = world_size * self.seq_len
        self.val_tokens = 10485760 if not is_a10g else 16*self.seq_len
        self.gradient_accumulation_steps = math.ceil(8 / world_size)
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps



def gen_name(model: GPT, args: TrainConfig, uuid: uuid.UUID):
    
    if args.use_mtp:
        if args.use_deepseek_mtp:
            mtp_type = "deepseek"
        else:
            mtp_type = "mtp"
    else:
        mtp_type = ""
    
    return (
        f'{model.__class__.__name__}_torch_compile={args.torch_compile}_{mtp_type}'
        f'_batch_size={args.batch_size}'
        f'_use_liger={args.use_liger}'
        f'_use_mtp={args.use_mtp}'
        f'_use_adam_mini={args.use_adam_mini}'
        f'_num_layers={args.num_layers}'
        f'_num_heads={args.num_heads}'
        f'_model_dim={args.model_dim}'
        f'_n_mtp_tokens={args.n_mtp_tokens}'
        f'_date={datetime.datetime.now().strftime("%Y-%m-%d")}'
    )

def print0(_out_dict : Union[dict, str], console=False, use_wandb=True):
    if master_process:
        if isinstance(_out_dict, dict):
            if use_wandb:
                wandb.log(_out_dict)
            
            s = ""
            for k, v in _out_dict.items():
                s += f"{k}: {v} "
        else:
            s = _out_dict
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)



 # begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)


def train(
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin",
    val_files: str = "data/fineweb10B/fineweb_val_*.bin",
    # optimization
    num_iterations: int = 1393,
    cooldown_frac: float = 0.4,
    # evaluation and logging
    val_loss_every: int = 125,
    # implementation
    seq_len: int = None,  # will be set based on is_a10g
    save_checkpoint: bool = False,
    use_liger: bool = True,
    use_adam_mini: bool = False,
    use_mtp: bool = False,
    num_layers: int = 12,
    n_mtp_tokens: int = 2,
    use_wandb: bool = True,
    torch_compile: bool = True,
    use_deepseek_mtp: bool = True,
    proj_fp8: bool = False,
    bfloat16: bool = False,
    BLOCK_SIZE: int = 128,
    IS_MODAL: bool = True
):
    # Create config object from individual arguments
    args = TrainConfig(
        train_files=train_files,
        val_files=val_files,
        num_iterations=num_iterations,
        cooldown_frac=cooldown_frac,
        val_loss_every=val_loss_every,
        seq_len=64*(1024) if not is_a10g else 16*(1024),  # Default based on is_a10g
        save_checkpoint=save_checkpoint,
        use_liger=use_liger,
        use_adam_mini=use_adam_mini,
        use_mtp=use_mtp,
        num_layers=num_layers,
        n_mtp_tokens=n_mtp_tokens,
        use_wandb=use_wandb,
        torch_compile=torch_compile,
        use_deepseek_mtp=use_deepseek_mtp,
        proj_fp8=proj_fp8,
        bfloat16=bfloat16,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_MODAL=IS_MODAL
    )
    
    if args.use_mtp:
        model = MTPGPT(
            vocab_size=50257,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            model_dim=args.model_dim,
            n_mtp_tokens=args.n_mtp_tokens,
            use_liger=args.use_liger,
            use_deepseek_mtp=args.use_deepseek_mtp,
            proj_fp8=args.proj_fp8
        ).cuda() 
    else:
        model = GPT(
            vocab_size=50257, 
            num_layers=args.num_layers, 
            num_heads=args.num_heads,
            model_dim=args.model_dim,
            use_fp8=args.proj_fp8
        ).cuda() 
        
    if args.use_wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = 'a3469eb2df23f67e4d6907ebacf50ffb4ee664f7'
        name = gen_name(model, args, run_id)
        wandb.init(
            project="modded-nanogpt", 
            name=name,
            config=args
        )



    if args.use_mtp:
        seq_len = seq_len + args.n_mtp_tokens # we add the n_tokens to the seq_len for the MTP

    # load data
    train_loader = distributed_data_generator(
        args.train_files, 
        args.batch_size,
        rank, 
        world_size,
        from_path='/root/data' if args.IS_MODAL else None
    )




    # begin by printing this file (the Python code)
    print0(code, console=False)
    print0("="*100, console=False)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}", console=True)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}", console=True)
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
    print0("="*100)


    from utils import numel_params_million
    print0(f"Model created with {numel_params_million(model)}M parameters", console=True)
    for m in model.modules():
        
        # we optionally train in bfloat16
        if isinstance(m, nn.Embedding) or args.bfloat16:
            m.bfloat16()
            
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    if not args.use_adam_mini:
        # collect the parameters to optimize
        hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
        embed_params = [p for n, p in model.named_parameters() if "embed" in n]
        scalar_params = [p for p in model.parameters() if p.ndim < 2]

        if isinstance(unwrap(model), MTPGPT):
            if isinstance(model.mtp, MTP):
                head_params = [head.weight for head in model.mtp.heads]
            else:
                head_params = [model.mtp.shared_head.weight]
                hidden_matrix_params.extend([p for n, p in chain(model.mtp.blocks.named_parameters(), model.mtp.projs.named_parameters()) if p.ndim >= 2 and "embed" not in n])
        else:
            head_params = [model.lm_head.weight]

        # init the optimizer(s)
        adam_params = [dict(params=head_params, lr=0.008), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
        # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
        # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
        optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
        optimizers = [optimizer1, optimizer2]
    else:
        from adam_mini import Adam_mini
        
        optimizers = [Adam_mini(
            model.named_parameters(), 
            lr=0.008, 
            model_sharding=False, 
            dim=args.model_dim, 
            n_heads=args.num_heads
            )
        ]
                
        

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        t = 1 - step / args.num_iterations # time remaining in training
        assert 1 >= t >= 0
        w = min(t / args.cooldown_frac, 1.0) # 1 -> 0
        return w * 1.0 + (1 - w) * 0.1
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
    @lru_cache(1)
    def sw_num_blks(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    if args.torch_compile:
        print0("Torch compile enabled, compiling model...", console=True)
        time_start = time.perf_counter()
        model = torch.compile(model)
        time_end = time.perf_counter()
        print0(f"Torch compile time: {time_end - time_start:.2f} seconds", console=True)
    else:
        print0("Torch compile disabled", console=True)
        
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_steps = args.num_iterations
    print0(f"Starting training loop for {train_steps} steps", console=True)
    total_tokens_seen = 0  # Track lifetime total tokens
    accum_loss = 0
    if args.use_mtp:
        accum_loss_dict = {}


    for step in range(train_steps + 1):
        print0(f"Beginning step {step}", console=True)
        last_step = (step == train_steps)
        
        if step == 10:
            print0("Resetting timing measurements", console=True)
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float("nan") if step <= 11 else (step - 10) + 1

        if step % args.gradient_accumulation_steps == 0:
            window_size = next_multiple_of_n((1728 * step) * args.gradient_accumulation_steps / train_steps, n=128)
            print0(f"Window size for step {step}: {window_size}", console=True)

        # Validation section
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            print0(f"Starting validation at step {step}", console=True)
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.seq_len
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(
                args.val_files, 
                val_batch_size, 
                rank, 
                world_size,
                from_path='/root/data' if args.IS_MODAL else None
            )
            val_loss = 0
            with torch.no_grad():
                for val_step in tqdm(range(val_steps), desc="Validation steps", leave=False, disable=not master_process):
                    x, y = next(val_loader)
                    if args.use_mtp:
                        _ ,loss_dict = model.forward(x, y, sw_num_blks(window_size), with_backward=False)
                        val_loss += loss_dict["loss_orig"] # this corresponds to the normal next token prediction loss
                    else:
                        val_loss += model.forward(x, y, sw_num_blks(window_size))
            
            print0("Validation complete, reducing loss", console=True)
            val_loss /= val_steps
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0({"step": step, "val_loss": val_loss, "train_time": training_time_ms, "train_time_minutes": training_time_ms / 60*1000}, console=True)
            if args.use_wandb:
                wandb.log({"step": step, "val_loss": val_loss, "train_time": training_time_ms})
            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            print0("Reached last step, breaking", console=True)
            break

        # Training section
        print0(f"Loading training batch for step {step}", console=True)
        inputs, targets = next(train_loader)
        batch_tokens = inputs.numel() * world_size
        total_tokens_seen += batch_tokens  # Lifetime total
        
        # Scale loss by gradient accumulation steps
        loss_scale = 1.0 / args.gradient_accumulation_steps
        
        
        if args.use_mtp:
            assert isinstance(model, MTPGPT)
            loss, loss_dict = model.forward(inputs, targets, sw_num_blks(window_size), with_backward=True)
            for k, v in loss_dict.items():
                if k in accum_loss_dict:
                    accum_loss_dict[k] += v * loss_scale
                else:
                    accum_loss_dict[k] = v * loss_scale

        else:
            loss = model.forward(inputs, targets, sw_num_blks(window_size))
            (loss_scale * loss).backward()
        
        accum_loss += loss * loss_scale           

        # Only perform optimizer step after accumulating gradients
        if (step + 1) % args.gradient_accumulation_steps == 0:
            print0("Reducing gradients", console=True)
            for param in model.parameters():
                if param.grad is not None:
                    if args.use_mtp:
                        # we multiply by loss_scale because we accumulated the loss over the gradient accumulation steps
                        grad = loss_scale * param.grad
                    else:
                        grad = param.grad
                        
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                    param.grad = grad
                
            print0("Updating optimizers", console=True)
            frac = min(step / 300, 1)
            if not args.use_adam_mini:
                for group in optimizer2.param_groups:
                    group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
                    
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
                
            model.zero_grad(set_to_none=True)
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"Completed step {step} in {approx_time:.2f} ms", console=True)
            print0(f"Loss: {accum_loss if not args.use_mtp else accum_loss_dict['loss_orig']}", console=True)
            
            tokens_per_sec = total_tokens_seen / (approx_time / 1000)
            print0({
                "step": step,
                "loss": accum_loss if not args.use_mtp else accum_loss_dict["loss_orig"],
                "tokens_seen": total_tokens_seen,  # Log total lifetime tokens
                "tokens_per_sec": tokens_per_sec,
                **(accum_loss_dict if args.use_mtp else {}),
            }, console=True)
            
            # reset accum_loss and accum_loss_dict
            accum_loss = 0
            if args.use_mtp:
                accum_loss_dict = {}


    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True
    )
    dist.destroy_process_group()



fire.Fire(train)
