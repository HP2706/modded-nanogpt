import inspect
from typing import Literal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import datetime
from tqdm import tqdm
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
from utils import next_multiple_of_n
from trainer_registry import resolve_adapter_by_type, load_adapter_from_path


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

def create_config_from_hydra(cfg: DictConfig):
    """Convert Hydra DictConfig to a simple namespace object with post-processing."""
    # Create a simple object to hold config values
    config = type('Config', (), {})()
    
    # Copy all config values (Hydra defaults merge everything into one flat structure)
    for key, value in cfg.items():
        setattr(config, key, value)
    
    # Post-processing logic (equivalent to __post_init__)
    config.num_heads = 6 if not is_a10g else 1  # NOTE there will be bugs if this is not 1 when model_dim != 768
    
    if config.type in ['deepseek-mtp', 'base-mtp']:
        assert hasattr(config, 'n_mtp_tokens') and config.n_mtp_tokens is not None, "n_mtp_tokens must be specified for deepseek-mtp or base-mtp"
        # First ensure base seq_len is multiple of 128
        config.seq_len = next_multiple_of_n(config.seq_len, n=128)
        # Then add the n_tokens
        config.seq_len = config.seq_len + config.n_mtp_tokens
    
    config.batch_size = world_size * config.seq_len
    config.val_tokens = 10485760 if not is_a10g else 16*config.seq_len
    config.gradient_accumulation_steps = math.ceil(8 / world_size)
    config.effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    
    return config



def gen_name(model: nn.Module, args, uuid: uuid.UUID):
    return (
        f'{model.__class__.__name__}_torch_compile={args.torch_compile}_{args.type}'
        f'_batch_size={args.batch_size}'
        f'_use_liger={args.use_liger}'
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


@hydra.main(version_base=None, config_path="configs")
def train(cfg: DictConfig):
    # Convert Hydra config to our expected format
    args = create_config_from_hydra(cfg)
    
    # Print config for debugging
    print0(f"Loaded configuration: {OmegaConf.to_yaml(cfg)}", console=True)

    # Resolve adapter and build model
    if args.model_adapter:
        adapter = load_adapter_from_path(args.model_adapter)
    else:
        adapter = resolve_adapter_by_type(args.type)

    # Validate/construct model cfg via adapter's Cfg if present
    model_cfg = None
    cfg_cls = getattr(adapter, 'Cfg', None)
    if cfg_cls is not None:
        if args.model_cfg is None:
            try:
                model_cfg = cfg_cls()
            except Exception:
                model_cfg = None
        else:
            model_cfg = args.model_cfg if isinstance(args.model_cfg, cfg_cls) else cfg_cls.model_validate(args.model_cfg)

    model = adapter.build(args, model_cfg)
        
    if args.use_wandb:
        import wandb
        os.environ["WANDB_API_KEY"] = 'a3469eb2df23f67e4d6907ebacf50ffb4ee664f7'
        name = gen_name(model, args, run_id)
        wandb.init(
            project="modded-nanogpt", 
            name=name,
            config=args
        )

    # load data
    data_path = getattr(args, 'modal_data_path', None) if args.IS_MODAL else getattr(args, 'local_data_path', None)
    train_loader = distributed_data_generator(
        args.train_files, 
        args.batch_size,
        rank, 
        world_size,
        from_path=data_path
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


    # Build optimizers/schedulers via adapter
    optimizers, schedulers = adapter.create_optimizers(
        model, args, rank=rank, world_size=world_size, device=device
    )
    
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
    accum_aux_logs: dict = {}


    for step in range(train_steps + 1):
        print0(f"Beginning step {step}", console=True)
        last_step = (step == train_steps)
        
        if step == 10:
            print0("Resetting timing measurements", console=True)
            training_time_ms = 0
            t0 = time.perf_counter()

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
                from_path=data_path
            )
            val_loss = 0
            with torch.no_grad():
                for val_step in tqdm(range(val_steps), desc="Validation steps", leave=False, disable=not master_process):
                    x, y = next(val_loader)
                    val_loss += adapter.val_step(model, x, y, sw_num_blks(window_size), args=args)
            
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
        
        loss, aux_logs = adapter.train_step(
            model, inputs, targets, sw_num_blks(window_size), loss_scale=loss_scale, args=args
        )
        for k, v in aux_logs.items():
            accum_aux_logs[k] = accum_aux_logs.get(k, 0) + v * loss_scale
        accum_loss += loss * loss_scale

        # Only perform optimizer step after accumulating gradients
        if (step + 1) % args.gradient_accumulation_steps == 0:
            print0("Reducing gradients", console=True)
            for param in model.parameters():
                if param.grad is not None:
                    if adapter.requires_scaled_grad_on_reduce():
                        grad = loss_scale * param.grad  # scale before all_reduce if model did internal backward
                    else:
                        grad = param.grad
                        
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                    param.grad = grad
                
            print0("Updating optimizers", console=True)
            frac = min(step / 300, 1)
            if not args.use_adam_mini:
                # Adjust momentum for Muon-like optimizers
                for opt in optimizers:
                    for group in getattr(opt, "param_groups", []):
                        if "momentum" in group:
                            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
                    
            for opt, sched in zip(optimizers, schedulers):
                opt.step()
                sched.step()
            
            adapter.post_optimizer_step(model, args=args)
                
            model.zero_grad(set_to_none=True)
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f"Completed step {step} in {approx_time:.2f} ms", console=True)
            display_loss = accum_aux_logs.get('loss_orig', accum_loss)
            print0(f"Loss: {display_loss}", console=True)
            
            tokens_per_sec = total_tokens_seen / (approx_time / 1000)
            print0({
                "step": step,
                "loss": display_loss,
                "tokens_seen": total_tokens_seen,  # Log total lifetime tokens
                "tokens_per_sec": tokens_per_sec,
                **accum_aux_logs,
            }, console=True)
            
            # reset accum_loss and accum_loss_dict
            accum_loss = 0
            accum_aux_logs = {}


    print0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        console=True
    )
    dist.destroy_process_group()

if __name__ == "__main__":
    train() 
