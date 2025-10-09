"""Utility modules to inject LoRA adapters into PyTorch models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from ..config import LoRAConfig


@dataclass
class LoRAState:
    """Keeps track of LoRA weights for a base layer."""

    A: nn.Parameter
    B: nn.Parameter
    alpha: float
    dropout: Optional[nn.Module]

    @property
    def scaling(self) -> float:
        return self.alpha / self.A.shape[0]


class LoRALinear(nn.Module):
    """Wrap a :class:`torch.nn.Linear` layer with a LoRA adapter."""

    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires a torch.nn.Linear base layer")
        self.base = base
        self.state = LoRAState(
            A=nn.Parameter(torch.zeros(config.rank, base.in_features)),
            B=nn.Parameter(torch.zeros(base.out_features, config.rank)),
            alpha=config.alpha,
            dropout=nn.Dropout(p=config.dropout) if config.dropout else None,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.state.A, a=math.sqrt(5))
        nn.init.zeros_(self.state.B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = self.base(input)
        x = input
        if self.state.dropout is not None:
            x = self.state.dropout(x)
        update = (x @ self.state.A.t()) @ self.state.B.t()
        return residual + self.state.scaling * update


def iter_named_linear_modules(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    """Yield all linear submodules of ``module``."""

    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child


def inject_lora(module: nn.Module, config: LoRAConfig) -> dict[str, LoRALinear]:
    """Replace ``nn.Linear`` modules with :class:`LoRALinear` wrappers.

    Parameters
    ----------
    module:
        Module whose ``Linear`` children should receive LoRA adapters.
    config:
        Configuration describing rank, scaling and dropout.

    Returns
    -------
    dict[str, LoRALinear]
        Mapping from fully-qualified module names to their LoRA wrappers.
    """

    replaced: dict[str, LoRALinear] = {}
    target_modules = config.target_modules

    for name, linear in iter_named_linear_modules(module):
        if target_modules and not any(token in name for token in target_modules):
            continue
        parent_name, _, attr_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        lora_layer = LoRALinear(linear, config)
        setattr(parent, attr_name, lora_layer)
        replaced[name] = lora_layer
    return replaced


def mark_only_lora_as_trainable(module: nn.Module) -> None:
    """Freeze the base model parameters and enable gradients for LoRA weights."""

    for param in module.parameters():
        param.requires_grad = False
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.base.weight.requires_grad = False
            if child.base.bias is not None:
                child.base.bias.requires_grad = False
            child.state.A.requires_grad = True
            child.state.B.requires_grad = True
            if child.state.dropout is not None:
                for param in child.state.dropout.parameters():
                    param.requires_grad = False

