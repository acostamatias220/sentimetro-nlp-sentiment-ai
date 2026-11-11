"""Classification metrics and error analysis.

Macro-F1 is the main metric: the ``neutro`` class is half the size of the
others and is the hardest one, so accuracy alone would hide a model that
ignores it. Every class weighs the same in macro-F1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# Pairs of classes whose confusion is a "polarity" error (much worse than neutral confusions).
POLAR_LABELS = ("negativo", "positivo")


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro/weighted F1, per-class report and confusion matrix."""
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    polar_errors = sum(
        int(cm[labels.index(a), labels.index(b)])
        for a in POLAR_LABELS
        for b in POLAR_LABELS
        if a != b and a in labels and b in labels
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "polar_error_rate": polar_errors / max(len(y_true), 1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "n": len(y_true),
    }


@dataclass(frozen=True)
class ErrorSample:
    text: str
    true: str
    pred: str
    confidence: float
    stars: int | None = None


def top_errors(
    texts: Sequence[str],
    y_true: Sequence[str],
    probs: NDArray[np.floating[Any]],
    labels: Sequence[str],
    *,
    n: int = 10,
    stars: Sequence[int] | None = None,
) -> list[ErrorSample]:
    """The ``n`` misclassified examples the model was most confident about.

    These are the most informative errors: label noise, sarcasm, mixed opinions
    ("el producto es bueno pero llegó tarde") or ratings that contradict the text.
    """
    pred_idx = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    true_idx = np.array([list(labels).index(y) for y in y_true])
    wrong = np.flatnonzero(pred_idx != true_idx)
    order = wrong[np.argsort(-confidence[wrong], kind="stable")][:n]
    return [
        ErrorSample(
            text=texts[i],
            true=labels[true_idx[i]],
            pred=labels[pred_idx[i]],
            confidence=float(confidence[i]),
            stars=int(stars[i]) if stars is not None else None,
        )
        for i in order
    ]


def summarize(metrics: dict[str, Any]) -> dict[str, float]:
    """Flat dict of scalar metrics, handy for MLflow and comparison tables."""
    flat = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "polar_error_rate": metrics["polar_error_rate"],
    }
    for label, values in metrics["per_class"].items():
        flat[f"f1_{label}"] = values["f1"]
    return flat
