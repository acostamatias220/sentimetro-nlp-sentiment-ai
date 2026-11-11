"""Confusion matrix plot (normalized by true class, with raw counts)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], labels: Sequence[str], title: str, path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(matrix, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    image = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(title, fontsize=11)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if norm[i, j] > 0.55 else "#1f2937"
            ax.text(
                j,
                i,
                f"{norm[i, j]:.0%}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
