"""CPU latency / throughput micro-benchmark for any predictor."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sentimetro.inference.predictor import Predictor


def directory_size_mb(path: Path, pattern: str) -> float:
    return sum(f.stat().st_size for f in path.glob(pattern)) / 1_048_576


def benchmark(
    predictor: Predictor, texts: Sequence[str], n_batches: int = 20, batch_size: int = 32
) -> dict[str, float]:
    """p50/p95 latency for single requests and throughput for batches."""
    texts = list(texts)
    predictor.predict_proba(texts[:batch_size])  # warm-up (lazy init, caches)

    single: list[float] = []
    for text in texts[: n_batches * 5]:
        started = time.perf_counter()
        predictor.predict_proba([text])
        single.append((time.perf_counter() - started) * 1000)

    total_texts = 0
    started = time.perf_counter()
    for i in range(n_batches):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        if not batch:
            break
        predictor.predict_proba(batch, batch_size=batch_size)
        total_texts += len(batch)
    elapsed = time.perf_counter() - started

    return {
        "latency_p50_ms": float(np.percentile(single, 50)),
        "latency_p95_ms": float(np.percentile(single, 95)),
        "throughput_texts_per_s": total_texts / elapsed if elapsed else 0.0,
    }
