"""High level wrapper combining compressor and decoder for PISCO."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..config import CompressorConfig, DecoderConfig
from .compressor import PISCOCompressor
from .decoder import PISCODecoder


class PISCOModel(nn.Module):
    def __init__(
        self,
        compressor_config: CompressorConfig,
        decoder_config: DecoderConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.compressor = PISCOCompressor(compressor_config, device=self.device)
        self.decoder = PISCODecoder(decoder_config, device=self.device)

    def forward(
        self,
        documents: list[str],
        queries: list[str],
        labels: Optional[list[str]] = None,
        max_document_length: Optional[int] = None,
    ) -> torch.Tensor:
        tokenized = self.compressor.tokenize(documents, max_length=max_document_length)
        compressed = self.compressor.forward(**tokenized)
        outputs = self.decoder(
            queries=queries,
            memory_tokens=compressed.memory_tokens,
            memory_attention_mask=compressed.attention_mask,
            labels=labels,
        )
        return outputs

    @torch.inference_mode()
    def answer(
        self,
        document: str,
        query: str,
        **generate_kwargs,
    ) -> str:
        compressed = self.compressor.compress([document])
        return self.decoder.generate(
            query=query,
            memory_tokens=compressed.memory_tokens[0],
            memory_attention_mask=compressed.attention_mask[0],
            **generate_kwargs,
        )

