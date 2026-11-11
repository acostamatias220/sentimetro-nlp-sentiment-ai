"""Backends that turn texts into class probabilities.

- ``onnx``: ONNX Runtime + the standalone ``tokenizers`` library. This is the
  production backend: no PyTorch needed, small Docker image, fast on CPU.
- ``sklearn``: the TF-IDF baseline pipeline (also used by the tests as a tiny
  dummy model, so CI never downloads large weights).
- ``transformers``: the PyTorch checkpoint, used for evaluation and to check
  ONNX parity (requires the ``train`` extra).

Every backend applies :func:`sentimetro.text.clean_text` first, exactly like
training did.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentimetro.model_card import ModelCard
from sentimetro.text import clean_text

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    scores: dict[str, float]


def softmax(logits: NDArray[Any]) -> FloatArray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    result: FloatArray = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
    return result


class Predictor(ABC):
    """Common interface: ``predict_proba`` returns ``(n_texts, n_labels)`` probabilities."""

    def __init__(self, card: ModelCard, model_dir: Path) -> None:
        self.card = card
        self.model_dir = model_dir
        self.labels = list(card.labels)

    @abstractmethod
    def _predict_proba(self, texts: list[str]) -> FloatArray: ...

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> FloatArray:
        cleaned = [clean_text(t) for t in texts]
        if not cleaned:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        chunks = [
            self._predict_proba(cleaned[i : i + batch_size])
            for i in range(0, len(cleaned), batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def predict(self, texts: list[str]) -> list[Prediction]:
        probs = self.predict_proba(texts)
        results = []
        for row in probs:
            best = int(row.argmax())
            results.append(
                Prediction(
                    label=self.labels[best],
                    score=float(row[best]),
                    scores={label: float(p) for label, p in zip(self.labels, row, strict=True)},
                )
            )
        return results


class SklearnPredictor(Predictor):
    def __init__(self, card: ModelCard, model_dir: Path) -> None:
        super().__init__(card, model_dir)
        import joblib

        self.pipeline = joblib.load(model_dir / card.artifact)
        classes = [str(c) for c in self.pipeline.classes_]
        # Reorder columns to the label order declared in the card.
        self._order = [classes.index(label) for label in self.labels]

    def _predict_proba(self, texts: list[str]) -> FloatArray:
        probs = np.asarray(self.pipeline.predict_proba(texts), dtype=np.float32)
        result: FloatArray = probs[:, self._order]
        return result


class OnnxPredictor(Predictor):
    def __init__(self, card: ModelCard, model_dir: Path) -> None:
        super().__init__(card, model_dir)
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if card.tokenizer is None or card.max_length is None:
            raise ValueError("ONNX model cards need `tokenizer` and `max_length`")
        self.tokenizer = Tokenizer.from_file(str(model_dir / card.tokenizer))
        self.tokenizer.enable_truncation(max_length=card.max_length)
        pad_token = "[PAD]" if self.tokenizer.token_to_id("[PAD]") is not None else "<pad>"
        pad_id = self.tokenizer.token_to_id(pad_token)
        self.tokenizer.enable_padding(pad_id=pad_id or 0, pad_token=pad_token)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(os.environ.get("SENTIMETRO_ORT_THREADS", "0"))
        if threads > 0:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / card.artifact), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _predict_proba(self, texts: list[str]) -> FloatArray:
        encodings = self.tokenizer.encode_batch(texts)
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        (logits,) = self.session.run(["logits"], feeds)
        return softmax(np.asarray(logits))


class TransformersPredictor(Predictor):
    def __init__(self, card: ModelCard, model_dir: Path) -> None:
        super().__init__(card, model_dir)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
        id2label = self.model.config.id2label
        self._order = [
            next(int(i) for i, name in id2label.items() if name == label) for label in self.labels
        ]

    def _predict_proba(self, texts: list[str]) -> FloatArray:
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.card.max_length,
            padding=True,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self.model(**enc).logits.numpy()
        result: FloatArray = softmax(logits)[:, self._order]
        return result


_BACKENDS: dict[str, type[Predictor]] = {
    "sklearn": SklearnPredictor,
    "onnx": OnnxPredictor,
    "transformers": TransformersPredictor,
}


def load_predictor(model_dir: str | Path) -> Predictor:
    """Read ``model_card.json`` in ``model_dir`` and build the matching backend."""
    directory = Path(model_dir)
    card = ModelCard.load(directory)
    return _BACKENDS[card.backend](card, directory)
