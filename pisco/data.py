"""Dataset helpers for training PISCO via sequence-level distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass
class QAExample:
    document: str
    question: str
    teacher_answer: str | None = None


class QADataset(Sequence[QAExample]):
    """Simple in-memory dataset used for prototyping and testing."""

    def __init__(self, examples: Iterable[QAExample]):
        self._examples = list(examples)

    def __getitem__(self, idx: int) -> QAExample:
        return self._examples[idx]

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[QAExample]:
        return iter(self._examples)

