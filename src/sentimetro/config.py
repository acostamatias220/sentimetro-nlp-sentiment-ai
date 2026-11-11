"""Typed, validated configuration loaded from YAML.

Every stage of the pipeline (data, baseline, transformer, export) reads its
parameters from ``configs/params.yaml``. Values can be overridden from the CLI
with dotted keys, e.g. ``--set transformer.epochs=3 data.seed=7``, so the same
code runs a quick CPU experiment or a full GPU training without edits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sentimetro import LABELS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    dataset_name: str = "SetFit/amazon_reviews_multi_es"
    # Pinned dataset commit so every run sees exactly the same data.
    dataset_revision: str = "16015418b488c9186fce74b058877ea939ca934d"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    # Original label is stars - 1 (0..4). Maps each star label to a class name.
    star_to_label: dict[int, str] = Field(
        default_factory=lambda: {
            0: "negativo",
            1: "negativo",
            2: "neutro",
            3: "positivo",
            4: "positivo",
        }
    )
    min_chars: int = Field(default=3, ge=1)
    max_chars: int = Field(default=5000, ge=10)
    seed: int = 42

    @field_validator("star_to_label")
    @classmethod
    def _labels_known(cls, value: dict[int, str]) -> dict[int, str]:
        unknown = set(value.values()) - set(LABELS)
        if unknown:
            raise ValueError(f"unknown labels {sorted(unknown)}; expected a subset of {LABELS}")
        if set(value) != {0, 1, 2, 3, 4}:
            raise ValueError("star_to_label must define the 5 star labels 0..4")
        return value


class TrackingConfig(_Strict):
    # Local MLflow backend: a SQLite file for runs/metrics + a folder for artifacts.
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "mlartifacts"
    experiment_name: str = "sentimetro"
    enabled: bool = True


class BaselineConfig(_Strict):
    output_dir: Path = Path("models/baseline")
    train_samples: int | None = Field(default=None, ge=100)
    word_ngram_max: int = Field(default=2, ge=1, le=3)
    char_ngram_min: int = Field(default=2, ge=1)
    char_ngram_max: int = Field(default=5, ge=2)
    max_features_word: int = Field(default=200_000, ge=1000)
    max_features_char: int = Field(default=300_000, ge=1000)
    min_df: int = Field(default=2, ge=1)
    c_grid: list[float] = Field(default_factory=lambda: [0.125, 0.25, 0.5, 1.0, 2.0])
    class_weight: Literal["balanced"] | None = "balanced"
    max_iter: int = Field(default=2000, ge=100)

    @model_validator(mode="after")
    def _check_ranges(self) -> BaselineConfig:
        if self.char_ngram_min > self.char_ngram_max:
            raise ValueError("char_ngram_min must be <= char_ngram_max")
        if not self.c_grid or any(c <= 0 for c in self.c_grid):
            raise ValueError("c_grid must contain positive values")
        return self


class TransformerConfig(_Strict):
    model_name: str = "dccuchile/distilbert-base-spanish-uncased"
    model_revision: str = "8b0f77825ae49a0d099bf5e3aea8da71f6c0851f"
    output_dir: Path = Path("models/transformer")
    # Subsampling keeps CPU training feasible; set to null to use everything (GPU).
    train_samples: int | None = Field(default=100_000, ge=100)
    eval_samples: int | None = Field(default=2_000, ge=100)
    max_length: int = Field(default=128, ge=16, le=512)
    epochs: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0, lt=1)
    train_batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int = Field(default=64, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    eval_steps: int = Field(default=500, ge=1)
    early_stopping_patience: int = Field(default=3, ge=1)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    # "balanced" weights the cross-entropy by inverse class frequency, which
    # helps the minority `neutro` class and therefore macro-F1.
    class_weighting: Literal["balanced"] | None = "balanced"
    fp16: bool = False
    num_threads: int | None = Field(default=None, ge=1)


class ExportConfig(_Strict):
    output_dir: Path = Path("models/production")
    source: Literal["transformer", "baseline"] = "transformer"
    opset: int = Field(default=18, ge=18)
    quantize: bool = True


class EvaluatedModel(_Strict):
    path: Path
    title: str


def _default_models() -> dict[str, EvaluatedModel]:
    return {
        "baseline": EvaluatedModel(path=Path("models/baseline"), title="TF-IDF + LogReg"),
        "distilbeto_onnx_int8": EvaluatedModel(
            path=Path("models/onnx-int8"), title="DistilBETO (ONNX int8)"
        ),
    }


class EvaluationConfig(_Strict):
    reports_dir: Path = Path("reports")
    # Models compared on the test split (missing folders are skipped with a warning).
    models: dict[str, EvaluatedModel] = Field(default_factory=_default_models)
    n_error_samples: int = Field(default=12, ge=1)
    benchmark_batches: int = Field(default=20, ge=1)


class Config(_Strict):
    data: DataConfig = DataConfig()
    tracking: TrackingConfig = TrackingConfig()
    baseline: BaselineConfig = BaselineConfig()
    transformer: TransformerConfig = TransformerConfig()
    export: ExportConfig = ExportConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def _parse_scalar(raw: str) -> Any:
    """Parse an override value with YAML rules (``3`` -> int, ``null`` -> None…)."""
    return yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides to a nested dict (returns a new dict)."""
    result: dict[str, Any] = yaml.safe_load(yaml.safe_dump(data)) or {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"invalid override {item!r}, expected key=value")
        node = result
        *parents, leaf = key.strip().split(".")
        for part in parents:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot override {key!r}: {part!r} is not a section")
            node = child
        node[leaf] = _parse_scalar(raw)
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load and validate a config file (defaults are used for missing keys)."""
    data: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data = apply_overrides(data, overrides)
    return Config.model_validate(data)
