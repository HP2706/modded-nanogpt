from typing import Protocol, Optional, Any, runtime_checkable
from torch import Tensor, nn
from jaxtyping import Float
from pydantic import BaseModel
from .LLMBase import ModelOutputMixin


class SequenceMixerConfig(BaseModel):
    """
    Pydantic config for sequence mixers (e.g., attention-like modules).
    Add additional common attributes as needed across implementations.
    """
    d_model: int

    class Config:
        arbitrary_types_allowed = True


class ChannelMixerConfig(BaseModel):
    """
    Pydantic config for channel mixers (e.g., MLP-like modules).
    Add additional common attributes as needed across implementations.
    """
    d_model: int
    d_mult: int

    class Config:
        arbitrary_types_allowed = True


class SequenceMixer(Protocol):
    """
    Protocol for sequence mixing operations (e.g., attention mechanisms).
    These components mix information across the sequence dimension.
    """

    def __init__(self, config: SequenceMixerConfig) -> None: ...

    def forward(
        self,
        x: Float[Tensor, "batch sequence_len d_model"],
        attention_mask: Optional[Tensor] = None,
    ) -> Float[Tensor, "batch sequence_len d_model"]: ...


class ChannelMixer(Protocol):
    """
    Protocol for channel mixing operations (e.g., MLPs).
    These components mix information across the feature/channel dimension.
    """

    def __init__(self, config: ChannelMixerConfig) -> None: ...

    def forward(
        self,
        x: Float[Tensor, "batch sequence_len d_model"],
    ) -> Float[Tensor, "batch sequence_len d_model"]: ...


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Minimal interface for models to plug into the training loop.
    Configs are Pydantic BaseModels (no Protocols for data containers).
    """

    cfg: BaseModel
    is_master_process: bool

    def __init__(self, cfg: BaseModel, is_master_process: bool) -> None: ...

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> ModelOutputMixin: ...

    def configure_optimizers(
        self,
        weight_decay: float,
        lr_config: Any,
        device_type: str,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> Any: ...

    def save_model_with_metadata(self, path: str, optimizer: Any, lr: float, step: int) -> None: ...

    def to(self, *args, **kwargs) -> nn.Module: ...
