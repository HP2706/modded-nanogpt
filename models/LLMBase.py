from abc import ABC, abstractmethod
from typing_extensions import Self
import bitsandbytes as bnb
from torch import nn, Tensor
import torch
import inspect
from typing import Optional, Type, Union, Callable, Protocol, Literal, overload, TypeVar
from Models.LLMS.configs import BaseTransformerConfig, ModelConfig, CONFIG_MAP
from pydantic import BaseModel, model_validator
from jaxtyping import Float, Int
from utils import LRConfig, count_non_embedding_params

T = TypeVar('T', bound='ModelMixin')

class ModelOutputMixin(BaseModel):
    logits : Tensor
    loss : Optional[Tensor] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='after')
    def check_tensor_values(self) -> 'Self':
        for field_name, value in self.model_dump().items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    raise ValueError(f"Tensor '{field_name}' contains NaN or Inf values.")
        return self


class ModelMixin(nn.Module, ABC):
    def __init__(
        self, 
        cfg : ModelConfig,
        is_master_process : bool
    ):
        super().__init__()
        assert isinstance(is_master_process, bool)
        self.is_master_process = is_master_process
        self.cfg = cfg

    def to(self, *args, **kwargs):
        device = kwargs.get('device')
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)
    
    def configure_optimizers(
        self, 
        weight_decay : float, 
        lr_config : LRConfig, 
        device_type : str,
        betas : tuple[float, float] = (0.9, 0.95)
    ):
        
        print("Non-embedding parameter count:", count_non_embedding_params(self))
        total_count = sum(p.numel() for p in self.parameters())
        print("Number of embedding params: ", total_count - count_non_embedding_params(self))
        print("Total parameter count:", total_count)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight Tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if self.is_master_process:
            print(f"num decayed parameter Tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter Tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if self.is_master_process:
            print(f"using fused AdamW: {use_fused}")

        optim_kwargs = {
            "lr": lr_config.max_lr,
            "betas": betas,
            "eps": 1e-8
        }
        if self.cfg.use_adam_8_bit:
            optimizer = bnb.optim.AdamW8bit(
                optim_groups, 
                **optim_kwargs
            )
        else:
            if lr_config.schedule_free:
                optimizer = AdamWScheduleFree(
                    optim_groups, 
                    **optim_kwargs
                )
            else:
                optimizer = torch.optim.AdamW(
                    optim_groups, 
                    **optim_kwargs,
                    fused=use_fused
                )
        return optimizer
    
    def save_model_with_metadata(
        self, 
        path: str, 
        optimizer: torch.optim.Optimizer, 
        lr : float,
        step: int
    ):
        assert self.cfg is not None, "model has to have attribute config"

        torch.save({
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': self.cfg if not isinstance(self.cfg, dict) else self.cfg.model_dump(),
            'config_type': self.cfg.__class__.__name__,
            'lr': lr,
            'step': step,
            'rng_state': torch.get_rng_state()
        }, path)


    @classmethod
    @overload
    def load_pretrained(
        cls: Type[T],
        model_path: str,
        is_master_process: bool,
        lr_config: LRConfig,
        with_metadata: Literal[False] = False,
    ) -> T: ...

    @classmethod
    @overload
    def load_pretrained(
        cls: Type[T],
        model_path: str,
        is_master_process: bool,
        lr_config: LRConfig,
        with_metadata: Literal[True] = True,
    ) -> tuple[
            T, 
            torch.optim.Optimizer, 
            float
        ]: ...

    @classmethod
    def load_pretrained(
        cls : Type[T], 
        model_path: str, 
        is_master_process: bool,
        lr_config : LRConfig,
        with_metadata: bool = False,
        ) -> Union[
            T, 
            tuple[
                T, 
                torch.optim.Optimizer, 
                float
            ]
        ]:
        state_dict = torch.load(model_path, map_location='cpu')
        config = state_dict['config']
        config_type = state_dict['config_type']
        config_cls = CONFIG_MAP[config_type]
        if isinstance(config, dict):
            config = config_cls(**config)
        model = cls(config, is_master_process)
        model.load_state_dict(state_dict['model'])
        
        if not with_metadata:
            return model
        else:
            optimizer = model.configure_optimizers(weight_decay=0.0, lr_config=lr_config, device_type="cpu")
            optimizer.load_state_dict(state_dict['optimizer'])
            rng_state = state_dict['rng_state']
            torch.set_rng_state(rng_state)
            lr = state_dict['lr']
            return (model, optimizer, lr)

    @abstractmethod
    def forward(
        self, 
        idx : Float[Tensor, "B T"], 
        targets : Optional[Int[Tensor, "B T"]] = None
    ) -> ModelOutputMixin:...

    @abstractmethod
    def _init_weights(self):...


class TransformerMixin(ModelMixin):
    def __init__(
        self, 
        cfg : BaseTransformerConfig,
        is_master_process : bool
    ):
        super().__init__(cfg, is_master_process) 
        if cfg.use_adam_8_bit:
            #this is adviced when using 8 bit optimizer
            self.embedding = bnb.nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_embed = bnb.nn.Embedding(cfg.n_ctx, cfg.d_model)
        else:
            self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)

        cfg.model_validate(cfg.model_dump())#this is for testing it is either instance or inherits
        self.cfg = cfg

    def _init_weights(self, module):
        #super()._init_weights(module)  TODO add this
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.cfg.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def check_forward(
        self, 
        idx : Float[Tensor, "B T"], 
        targets : Optional[Int[Tensor, "B T"]] = None
    ):
        T = idx.shape[1]
        assert T <= self.cfg.n_ctx, f"Cannot forward sequence of length {T}, block size is only {self.cfg.n_ctx}"
        if torch.any(idx >= self.cfg.vocab_size):
            raise ValueError(f"Index out of range. Max index should be {self.cfg.vocab_size - 1} got {idx.max()}")
        if targets is not None:
            assert targets.shape[0] == idx.shape[0], f"Targets and input idx have to have the same batch size, got {targets.shape[0]} and {idx.shape[0]}"
            