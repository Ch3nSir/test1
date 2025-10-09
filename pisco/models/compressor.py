"""Document compressor that produces memory tokens for PISCO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..config import CompressorConfig
from .lora import inject_lora, mark_only_lora_as_trainable


@dataclass
class CompressorOutput:
    memory_tokens: torch.Tensor
    attention_mask: torch.Tensor


class PISCOCompressor(nn.Module):
    """Encode long documents into a fixed set of memory tokens."""

    def __init__(self, config: CompressorConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self.backbone = AutoModel.from_pretrained(config.model_name_or_path)
        inject_lora(self.backbone, config.lora)
        mark_only_lora_as_trainable(self.backbone)

        hidden_size = self.backbone.config.hidden_size
        projection_size = config.projection_hidden_size or hidden_size
        self.pool = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.Tanh(),
            nn.Linear(projection_size, config.memory_tokens * hidden_size),
        )
        self.memory_tokens = config.memory_tokens
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def tokenize(self, texts: list[str], max_length: Optional[int] = None) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length or self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CompressorOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        pooled = self.pool(hidden)
        batch_size, _, hidden_size = hidden.shape
        memory = pooled.view(batch_size, self.memory_tokens, hidden_size)
        mem_mask = torch.ones(batch_size, self.memory_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        return CompressorOutput(memory_tokens=memory, attention_mask=mem_mask)

    @torch.inference_mode()
    def compress(self, documents: list[str], max_length: Optional[int] = None) -> CompressorOutput:
        tokenized = self.tokenize(documents, max_length=max_length)
        return self.forward(**tokenized)

