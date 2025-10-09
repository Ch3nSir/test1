"""Configuration dataclasses for the PISCO implementation.

This module defines reusable configuration containers for different
components of the PISCO framework.  They describe the compressor,
decoder and training hyper-parameters exposed in the paper and allows
scriptable construction from JSON/YAML (via ``dataclasses.asdict``) if
needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    Attributes
    ----------
    rank:
        Low rank of the adapter matrices (``r`` in the paper).
    alpha:
        Scaling factor used to match the magnitude of the base weights.
    dropout:
        Dropout probability applied to the LoRA update.
    target_modules:
        Optional tuple of module name substrings that should receive LoRA
        adapters.  ``None`` applies adapters to every linear layer.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: Optional[tuple[str, ...]] = None


@dataclass
class CompressorConfig:
    """Configuration options for :class:`~pisco.models.PISCOCompressor`."""

    model_name_or_path: str
    memory_tokens: int = 32
    projection_hidden_size: Optional[int] = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class DecoderConfig:
    """Configuration for :class:`~pisco.models.PISCODecoder`."""

    model_name_or_path: str
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    max_new_tokens: int = 256


@dataclass
class DistillationConfig:
    """Top-level configuration for sequence-level knowledge distillation."""

    compressor: CompressorConfig
    decoder: DecoderConfig
    teacher_model_name_or_path: str
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10_000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: Optional[str] = "bf16"
    batch_size: int = 4

