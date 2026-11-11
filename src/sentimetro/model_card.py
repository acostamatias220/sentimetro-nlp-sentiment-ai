"""``model_card.json``: metadata that travels with every exported model.

The serving layer only needs this file to know which backend to load, the
label order and the max sequence length; ``/model-info`` exposes it as-is.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sentimetro.utils import read_json, write_json

CARD_FILE = "model_card.json"
Backend = Literal["sklearn", "onnx", "transformers"]


class ModelCard(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str
    version: str
    backend: Backend
    labels: list[str]
    artifact: str
    tokenizer: str | None = None
    max_length: int | None = None
    base_model: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    git_commit: str | None = None
    mlflow_run_id: str | None = None
    training: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)

    def save(self, directory: Path) -> Path:
        path = directory / CARD_FILE
        write_json(path, self.model_dump(mode="json"))
        return path

    @classmethod
    def load(cls, directory: Path) -> ModelCard:
        path = directory / CARD_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return cls.model_validate(read_json(path))
