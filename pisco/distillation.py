"""Sequence-level knowledge distillation loop for PISCO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from .config import DistillationConfig
from .data import QAExample
from .models.pisco_model import PISCOModel


def collate_examples(examples: list[QAExample]) -> dict[str, list[str]]:
    documents = [ex.document for ex in examples]
    questions = [ex.question for ex in examples]
    answers = [ex.teacher_answer for ex in examples if ex.teacher_answer is not None]
    labels = answers if len(answers) == len(examples) else None
    return {"documents": documents, "questions": questions, "labels": labels}


@dataclass
class DistillationState:
    global_step: int
    best_loss: float


class SequenceDistiller:
    """Run sequence-level distillation with a teacher model."""

    def __init__(
        self,
        config: DistillationConfig,
        dataset: Iterable[QAExample],
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.dataset = list(dataset)
        self.pisco = PISCOModel(config.compressor, config.decoder, device=self.device)
        self.teacher = AutoModelForCausalLM.from_pretrained(config.teacher_model_name_or_path).to(self.device)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name_or_path)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        self.optimizer = AdamW(
            (p for p in self.pisco.parameters() if p.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        steps_per_epoch = max(1, len(self.dataset) // config.batch_size)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max(config.max_steps, steps_per_epoch),
        )
        self.grad_accum = config.gradient_accumulation_steps
        self.max_grad_norm = config.max_grad_norm
        self.mixed_precision = config.mixed_precision

    def _prepare_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_examples,
        )

    @torch.no_grad()
    def _generate_teacher_labels(self, documents: list[str], questions: list[str]) -> list[str]:
        prompts = [f"{q}\n\n{d}" for q, d in zip(questions, documents)]
        tokenized = self.teacher_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.teacher.generate(
            **tokenized,
            max_new_tokens=self.config.decoder.max_new_tokens,
            pad_token_id=self.teacher_tokenizer.pad_token_id,
        )
        generated = []
        prompt_lengths = tokenized.attention_mask.sum(dim=1)
        for seq, prompt_len in zip(outputs, prompt_lengths):
            prompt_len = int(prompt_len.item())
            answer = seq[prompt_len:]
            generated.append(self.teacher_tokenizer.decode(answer, skip_special_tokens=True))
        return generated

    def train(self) -> DistillationState:
        dataloader = self._prepare_dataloader()
        scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision is not None)
        global_step = 0
        best_loss = float("inf")
        self.pisco.train()

        for epoch in range(max(1, self.config.max_steps // max(1, len(dataloader)) + 1)):
            for batch in dataloader:
                documents = batch["documents"]
                questions = batch["questions"]
                labels = batch["labels"]
                if labels is None:
                    labels = self._generate_teacher_labels(documents, questions)

                with torch.cuda.amp.autocast(
                    enabled=self.mixed_precision is not None,
                    dtype=torch.bfloat16 if self.mixed_precision == "bf16" else torch.float16,
                ):
                    outputs = self.pisco(
                        documents=documents,
                        queries=questions,
                        labels=labels,
                    )
                    loss = outputs.loss

                loss = loss / self.grad_accum
                scaler.scale(loss).backward()

                if (global_step + 1) % self.grad_accum == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.pisco.parameters(), self.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
                best_loss = min(best_loss, loss.item())
                if global_step >= self.config.max_steps:
                    return DistillationState(global_step=global_step, best_loss=best_loss)
        return DistillationState(global_step=global_step, best_loss=best_loss)

