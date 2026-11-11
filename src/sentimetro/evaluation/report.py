"""Evaluate every trained model on the held-out test split and write ``reports/``.

Outputs (small, versioned in git so results are reviewable without retraining):

- ``reports/metrics.json``: all metrics, benchmarks and model sizes.
- ``reports/comparison.md``: the comparison table used in the README.
- ``reports/<model>/``: confusion matrix (PNG), per-class report and the most
  confident errors for qualitative error analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sentimetro import LABELS
from sentimetro.config import Config
from sentimetro.data.preprocess import load_split
from sentimetro.evaluation.benchmark import benchmark, directory_size_mb
from sentimetro.evaluation.metrics import compute_metrics, summarize, top_errors
from sentimetro.evaluation.plots import plot_confusion_matrix
from sentimetro.inference.predictor import Predictor, load_predictor
from sentimetro.model_card import ModelCard
from sentimetro.tracking import start_run
from sentimetro.utils import write_json

logger = logging.getLogger(__name__)

WEIGHT_PATTERNS = ("*.onnx", "*.safetensors", "*.joblib")


def _model_size(model_dir: Path) -> float:
    return sum(directory_size_mb(model_dir, p) for p in WEIGHT_PATTERNS)


def _per_class_table(metrics: dict[str, Any]) -> str:
    rows = ["| Clase | Precisión | Recall | F1 | Soporte |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, v in metrics["per_class"].items():
        rows.append(
            f"| {label} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} "
            f"| {v['support']} |"
        )
    rows.append(f"| **macro** | | | **{metrics['macro_f1']:.3f}** | {metrics['n']} |")
    return "\n".join(rows)


def _escape(text: str, limit: int = 220) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_model_report(
    name: str, title: str, metrics: dict[str, Any], errors: list[Any], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["labels"], title, out_dir / "confusion_matrix.png"
    )
    lines = [
        f"# {title}",
        "",
        f"Accuracy **{metrics['accuracy']:.4f}** · macro-F1 **{metrics['macro_f1']:.4f}** · "
        f"errores de polaridad (negativo ↔ positivo) **{metrics['polar_error_rate']:.2%}** "
        f"sobre {metrics['n']} reseñas de test.",
        "",
        _per_class_table(metrics),
        "",
        "![Matriz de confusión](confusion_matrix.png)",
        "",
        "## Errores con mayor confianza",
        "",
        "Los errores en los que el modelo estaba más seguro: suelen ser ruido de etiqueta "
        "(estrellas que no coinciden con el texto), opiniones mixtas o ironía.",
        "",
        "| Estrellas | Real | Predicción | Confianza | Texto |",
        "| :---: | --- | --- | ---: | --- |",
    ]
    lines += [
        f"| {e.stars} | {e.true} | {e.pred} | {e.confidence:.2f} | {_escape(e.text)} |"
        for e in errors
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "metrics.json", metrics)
    logger.info("  report -> %s", out_dir)


def comparison_table(results: dict[str, dict[str, Any]]) -> str:
    header = (
        "| Modelo | Train | Accuracy | Macro-F1 | F1 neutro | Err. polaridad | "
        "Latencia p50 (1 texto) | Throughput (lotes de 32) | Tamaño |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = [header]
    for r in results.values():
        s = r["summary"]
        b = r["benchmark"]
        rows.append(
            f"| {r['title']} | {r['train_rows'] / 1000:.0f}k | {s['accuracy']:.4f} "
            f"| **{s['macro_f1']:.4f}** | "
            f"{s['f1_neutro']:.4f} | {s['polar_error_rate']:.2%} | {b['latency_p50_ms']:.1f} ms | "
            f"{b['throughput_texts_per_s']:.0f} textos/s | {r['size_mb']:.0f} MB |"
        )
    return "\n".join(rows)


def _train_rows(card: ModelCard) -> int:
    return int(card.training.get("train_rows", 0))


def evaluate_model(
    predictor: Predictor, test: pd.DataFrame, n_errors: int
) -> tuple[dict[str, Any], list[Any]]:
    probs = predictor.predict_proba(test["text"].tolist())
    y_pred = [predictor.labels[i] for i in probs.argmax(axis=1)]
    metrics = compute_metrics(test["label"].tolist(), y_pred, LABELS)
    errors = top_errors(
        test["text"].tolist(),
        test["label"].tolist(),
        probs,
        predictor.labels,
        n=n_errors,
        stars=test["stars"].tolist(),
    )
    return metrics, errors


def evaluate_all(cfg: Config) -> dict[str, dict[str, Any]]:
    test = load_split(cfg.data, "test")
    reports = cfg.evaluation.reports_dir
    results: dict[str, dict[str, Any]] = {}

    with start_run(cfg.tracking, "evaluation-test") as run:
        for key, entry in cfg.evaluation.models.items():
            model_dir = Path(entry.path)
            if not (model_dir / "model_card.json").exists():
                logger.warning("Skipping %s: %s not found", key, model_dir)
                continue
            logger.info("Evaluating %s (%s)", key, model_dir)
            predictor = load_predictor(model_dir)
            metrics, errors = evaluate_model(predictor, test, cfg.evaluation.n_error_samples)
            bench = benchmark(predictor, test["text"].tolist(), cfg.evaluation.benchmark_batches)
            summary = summarize(metrics)
            results[key] = {
                "title": entry.title,
                "model_dir": str(model_dir),
                "train_rows": _train_rows(predictor.card),
                "summary": summary,
                "benchmark": bench,
                "size_mb": _model_size(model_dir),
            }
            write_model_report(key, entry.title, metrics, errors, reports / key)
            run.log_metrics({f"{key}.test_{k}": v for k, v in summary.items()})
            run.log_metrics({f"{key}.{k}": v for k, v in bench.items()})
            logger.info(
                "  macro-F1 %.4f · accuracy %.4f · p50 %.1f ms",
                summary["macro_f1"],
                summary["accuracy"],
                bench["latency_p50_ms"],
            )

        write_json(reports / "metrics.json", results)
        (reports / "comparison.md").write_text(comparison_table(results) + "\n", encoding="utf-8")
        run.log_artifact(reports, artifact_path="reports")
        _update_production_card(cfg, results)
    return results


def _update_production_card(cfg: Config, results: dict[str, dict[str, Any]]) -> None:
    """Copy the test metrics of the published model into the production card."""
    production = cfg.export.output_dir
    if not (production / "model_card.json").exists():
        return
    card = ModelCard.load(production)
    source = Path(str(card.training.get("published_from", "")))
    for result in results.values():
        if Path(result["model_dir"]) == source:
            card = card.model_copy(update={"test_metrics": result["summary"]})
            card.save(production)
            logger.info("Updated %s with test metrics", production / "model_card.json")
            return
