"""Decoder that consumes memory tokens and answers questions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DecoderConfig
from .lora import inject_lora, mark_only_lora_as_trainable


@dataclass
class DecoderBatch:
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]


class MemoryPrefixProjector(nn.Module):
    """Project memory tokens into the embedding space for prefix-tuning."""

    def __init__(self, hidden_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, memory_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(memory_tokens)


class PISCODecoder(nn.Module):
    def __init__(self, config: DecoderConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        inject_lora(self.model, config.lora)
        mark_only_lora_as_trainable(self.model)
        self.memory_projector = MemoryPrefixProjector(
            hidden_size=self.model.config.hidden_size,
            embed_dim=self.model.get_input_embeddings().embedding_dim,
        )
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def prepare_batch(
        self,
        queries: list[str],
        memory_tokens: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        labels: Optional[list[str]] = None,
    ) -> tuple[DecoderBatch, torch.Tensor]:
        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        projected_memory = self.memory_projector(memory_tokens)
        mem_mask = memory_attention_mask.to(self.device)

        if labels is not None:
            tokenized_answers = self.tokenizer(
                labels,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            eos = torch.full(
                (tokenized_answers.input_ids.size(0), 1),
                self.tokenizer.eos_token_id,
                device=self.device,
                dtype=torch.long,
            )
            answer_ids = torch.cat([tokenized_answers.input_ids, eos], dim=1)
            ones = torch.ones_like(answer_ids)
            combined_ids = torch.cat([tokenized_queries.input_ids, answer_ids], dim=1)
            combined_attention = torch.cat([tokenized_queries.attention_mask, ones], dim=1)
            labels_tensor = torch.full_like(combined_ids, -100)
            labels_tensor[:, tokenized_queries.input_ids.shape[1]:] = answer_ids
            input_embeds = self.model.get_input_embeddings()(combined_ids)
        else:
            combined_attention = tokenized_queries.attention_mask
            labels_tensor = None
            input_embeds = self.model.get_input_embeddings()(tokenized_queries.input_ids)

        input_embeds = torch.cat([projected_memory, input_embeds], dim=1)
        attention_mask = torch.cat([mem_mask, combined_attention], dim=1)

        batch = DecoderBatch(attention_mask=attention_mask, labels=labels_tensor)
        return batch, input_embeds

    def forward(
        self,
        queries: list[str],
        memory_tokens: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        labels: Optional[list[str]] = None,
    ) -> torch.Tensor:
        batch, input_embeds = self.prepare_batch(queries, memory_tokens, memory_attention_mask, labels)
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        return outputs

    @torch.inference_mode()
    def generate(
        self,
        query: str,
        memory_tokens: torch.Tensor,
        memory_attention_mask: torch.Tensor,
        **generate_kwargs,
    ) -> str:
        projected_memory = self.memory_projector(memory_tokens.unsqueeze(0))
        mem_mask = memory_attention_mask.unsqueeze(0).to(self.device)
        tokenized = self.tokenizer(query, return_tensors="pt").to(self.device)
        input_embeds = self.model.get_input_embeddings()(tokenized.input_ids)
        input_embeds = torch.cat([projected_memory, input_embeds], dim=1)
        attention_mask = torch.cat([mem_mask, tokenized.attention_mask], dim=1)
        output_ids = self.model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )
        generated = output_ids[0]
        generated = generated[tokenized.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

