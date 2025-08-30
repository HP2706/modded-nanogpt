from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple, Optional, Any, List

import torch
from torch import nn, Tensor

from optimizer import Muon


class ModelAdapter(Protocol):
    """Protocol for adapting models to the trainer.

    Implementations encapsulate model construction, optimizer grouping,
    and per-step train/val behavior so the trainer stays generic.
    """

    def build(self, args, cfg: Optional[Any]) -> nn.Module: ...

    def create_optimizers(
        self,
        model: nn.Module,
        args,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> Tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]: ...

    def train_step(
        self,
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        sw_num_blks: Tensor,
        *,
        loss_scale: float,
        args,
    ) -> Tuple[Tensor, Dict[str, Tensor]]: ...

    def val_step(
        self,
        model: nn.Module,
        inputs: Tensor,
        targets: Tensor,
        sw_num_blks: Tensor,
        *,
        args,
    ) -> Tensor: ...

    def requires_scaled_grad_on_reduce(self) -> bool:
        """Whether to multiply grads by loss_scale before all_reduce.

        MTP models do internal backward without loss scaling; we compensate
        during gradient reduction. Most others return a loss and we scale that
        before backward, so no extra grad scaling is needed.
        """
        ...

    def post_optimizer_step(self, model: nn.Module, *, args) -> None: ...


def _default_group_params_for_gpt_like(model: nn.Module):
    """Parameter grouping shared by GPT-like architectures.

    - hidden_matrix_params: blocks parameters with ndim >= 2 (excluding embeds)
    - embed_params: any parameter whose name contains 'embed'
    - scalar_params: parameters with ndim < 2
    - head_params: language modeling head weight(s)
    """
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]
    return hidden_matrix_params, embed_params, scalar_params, head_params


def _adam_muon_optimizers(
    *,
    hidden_matrix_params: list[nn.Parameter],
    embed_params: list[nn.Parameter],
    scalar_params: list[nn.Parameter],
    head_params: list[nn.Parameter],
    rank: int,
    world_size: int,
    device: torch.device,
):
    adam_params = [
        dict(params=head_params, lr=0.008),
        dict(params=embed_params, lr=0.6),
        dict(params=scalar_params, lr=0.04),
    ]
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size, device=device)
    return optimizer1, optimizer2


def _maybe_adam_mini(model: nn.Module, *, args):
    if not getattr(args, "use_adam_mini", False):
        return None
    from adam_mini import Adam_mini
    opt = Adam_mini(
        model.named_parameters(),
        lr=0.008,
        model_sharding=False,
        dim=args.model_dim,
        n_heads=args.num_heads,
    )
    return [opt]


def make_schedulers(optimizers: List[torch.optim.Optimizer], args):
    """Create standard cool-down LR schedulers for provided optimizers."""
    def get_lr(step: int):
        t = 1 - step / args.num_iterations
        w = min(t / args.cooldown_frac, 1.0)
        return w * 1.0 + (1 - w) * 0.1
    return [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]


@dataclass
class DefaultAdapter:
    """Default adapter with common behaviors.

    - Standard train/val steps for models that accept (idx, targets, sw_num_blks)
    - No-op post step and no grad scaling by default
    - Helper to build schedulers
    Subclasses can override any method as needed.
    """

    Cfg: Any | None = None  # subclasses may set to a Pydantic BaseModel

    def create_schedulers(self, optimizers: List[torch.optim.Optimizer], args):
        return make_schedulers(optimizers, args)

    def train_step(self, model, inputs, targets, sw_num_blks, *, loss_scale, args):
        loss = model.forward(inputs, targets, sw_num_blks)
        (loss_scale * loss).backward()
        return loss, {}

    def val_step(self, model, inputs, targets, sw_num_blks, *, args):
        return model.forward(inputs, targets, sw_num_blks)

    def requires_scaled_grad_on_reduce(self) -> bool:
        return False

    def post_optimizer_step(self, model: nn.Module, *, args) -> None:
        pass


def resolve_adapter_by_type(model_type: str) -> ModelAdapter:
    path = ADAPTER_PATHS.get(model_type)
    if path is None:
        raise ValueError(f"Invalid model type: {model_type}")
    return load_adapter_from_path(path)


def load_adapter_from_path(dotted_path: str) -> ModelAdapter:
    """Dynamically load an adapter given a dotted import path.

    Accepts forms like: "my_pkg.my_mod:MyAdapter" or "my_pkg.my_mod.MyAdapter".
    The resulting object can be either an instance or a class with no-arg ctor.
    """
    import importlib

    if ":" in dotted_path:
        module_path, obj_name = dotted_path.split(":", 1)
    else:
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid adapter path: {dotted_path}")
        module_path, obj_name = parts
    mod = importlib.import_module(module_path)
    obj = getattr(mod, obj_name)
    return obj() if callable(obj) and isinstance(obj, type) else obj


# Canonical adapter locations inside model files
ADAPTER_PATHS: Dict[str, str] = {
    "current-best": "models.Current_best_gpt:CurrentBestAdapter",
    "ngpt": "models.ngpt:NGPTAdapter",
    "deepseek-mtp": "models.mtp_model:MTPAdapter",
    "base-mtp": "models.mtp_model:MTPAdapter",
    "nsa": "models.Nsa:NSAAdapter",
    "gpt2": "models.gpt2:GPT2Adapter",
    "sedd": "models.SEDD:SEDDAdapter",
    # New model adapters from our recent fixes
    "mamba": "models.Mamba:MambaAdapter",
    "megabyte": "models.MegaByte:MegaByteAdapter", 
    "mixture-of-depths": "models.MixtureOfDepths:MoDAdapter",
    "mixture-of-experts": "models.MixtureOfExperts:MoEAdapter",
    "mdlm": "models.MDLM:MDLMAdapter",
    "d3pm": "models.D3PM:D3PMAdapter",
    "recurrent-depth": "models.RecurrentDepth:RecurrentDepthAdapter",
    "vanilla": "models.VanillaTransformer:VanillaAdapter",
    "ttt": "models.ttt:TTTAdapter",
    "titans": "models.titans:TitansAdapter",
}
