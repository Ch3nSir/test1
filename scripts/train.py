"""Command line entry point for training the PISCO compressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pisco.config import CompressorConfig, DecoderConfig, DistillationConfig, LoRAConfig
from pisco.data import QAExample, QADataset
from pisco.distillation import SequenceDistiller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PISCO via sequence-level distillation")
    parser.add_argument("config", type=Path, help="Path to a JSON configuration file")
    parser.add_argument("--output", type=Path, default=Path("pisco_checkpoint.pt"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset", type=Path, help="Path to a JSONL dataset file")
    return parser.parse_args()


def load_config(path: Path) -> DistillationConfig:
    data = json.loads(path.read_text())
    compressor = CompressorConfig(
        model_name_or_path=data["compressor"]["model_name_or_path"],
        memory_tokens=data["compressor"].get("memory_tokens", 32),
        projection_hidden_size=data["compressor"].get("projection_hidden_size"),
        lora=LoRAConfig(**data["compressor"].get("lora", {})),
    )
    decoder = DecoderConfig(
        model_name_or_path=data["decoder"]["model_name_or_path"],
        lora=LoRAConfig(**data["decoder"].get("lora", {})),
        max_new_tokens=data["decoder"].get("max_new_tokens", 256),
    )
    cfg = DistillationConfig(
        compressor=compressor,
        decoder=decoder,
        teacher_model_name_or_path=data["teacher_model_name_or_path"],
        learning_rate=data.get("learning_rate", 1e-4),
        weight_decay=data.get("weight_decay", 0.01),
        warmup_steps=data.get("warmup_steps", 100),
        max_steps=data.get("max_steps", 10_000),
        gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
        max_grad_norm=data.get("max_grad_norm", 1.0),
        mixed_precision=data.get("mixed_precision", "bf16"),
        batch_size=data.get("batch_size", 4),
    )
    return cfg


def load_dataset(path: Path) -> QADataset:
    examples: list[QAExample] = []
    with path.open() as handle:
        for line in handle:
            payload: dict[str, Any] = json.loads(line)
            examples.append(
                QAExample(
                    document=payload["document"],
                    question=payload["question"],
                    teacher_answer=payload.get("teacher_answer"),
                )
            )
    return QADataset(examples)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = load_dataset(args.dataset)
    distiller = SequenceDistiller(config, dataset, device=torch.device(args.device))
    state = distiller.train()
    torch.save({"model_state": distiller.pisco.state_dict(), "state": state.__dict__}, args.output)


if __name__ == "__main__":
    main()

