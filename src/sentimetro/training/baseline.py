"""Classic baseline: TF-IDF (word + char n-grams) + logistic regression.

Why this baseline:
- Trains on the *full* 200k reviews in minutes on a laptop CPU.
- Character n-grams (2-5, within word boundaries) are robust to typos, missing
  accents and morphology ("buenísimo", "buenisimo", "buenooo").
- Logistic regression gives calibrated-ish probabilities and is fully explainable
  (the coefficients are the most positive / negative n-grams).

The regularization strength ``C`` is chosen on the validation split by macro-F1.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from sentimetro import LABELS, __version__
from sentimetro.config import Config
from sentimetro.data.preprocess import load_split, stratified_sample
from sentimetro.evaluation.metrics import compute_metrics, summarize
from sentimetro.model_card import ModelCard
from sentimetro.tracking import start_run
from sentimetro.utils import git_commit, set_seed

logger = logging.getLogger(__name__)

ARTIFACT = "pipeline.joblib"


def build_pipeline(cfg: Config, c: float) -> Pipeline:
    b = cfg.baseline
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, b.word_ngram_max),
                    min_df=b.min_df,
                    max_features=b.max_features_word,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",  # keep 1-char tokens like "y", "a", "5"
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="char_wb",
                    ngram_range=(b.char_ngram_min, b.char_ngram_max),
                    min_df=b.min_df,
                    max_features=b.max_features_char,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    clf = LogisticRegression(
        C=c,
        class_weight=b.class_weight,
        max_iter=b.max_iter,
        random_state=cfg.data.seed,
    )
    return Pipeline([("features", features), ("clf", clf)])


def top_features(pipeline: Pipeline, k: int = 15) -> dict[str, list[str]]:
    """Most influential n-grams per class (for the README / sanity checks)."""
    names = pipeline.named_steps["features"].get_feature_names_out()
    clf: LogisticRegression = pipeline.named_steps["clf"]
    result: dict[str, list[str]] = {}
    for idx, label in enumerate(clf.classes_):
        best = clf.coef_[idx].argsort()[::-1][:k]
        result[str(label)] = [str(names[i]).split("__", 1)[-1] for i in best]
    return result


def train_baseline(cfg: Config, output_dir: Path | None = None) -> ModelCard:
    set_seed(cfg.data.seed)
    out = output_dir or cfg.baseline.output_dir
    out.mkdir(parents=True, exist_ok=True)

    train = stratified_sample(
        load_split(cfg.data, "train"), cfg.baseline.train_samples, cfg.data.seed
    )
    val = load_split(cfg.data, "validation")
    logger.info("Baseline: %d train / %d validation reviews", len(train), len(val))

    params: dict[str, Any] = cfg.baseline.model_dump(mode="json", exclude={"output_dir"})
    params["train_rows"] = len(train)

    with start_run(cfg.tracking, "baseline-tfidf-logreg") as run:
        run.set_tags({"model_type": "baseline", "git_commit": git_commit()})
        run.log_params(params)

        best: tuple[float, float, Pipeline] | None = None
        for c in cfg.baseline.c_grid:
            started = time.perf_counter()
            pipeline = build_pipeline(cfg, c).fit(train["text"], train["label"])
            elapsed = time.perf_counter() - started
            preds = pipeline.predict(val["text"])
            macro_f1 = compute_metrics(val["label"].tolist(), preds, LABELS)["macro_f1"]
            logger.info("  C=%-5g macro-F1=%.4f (%.0fs)", c, macro_f1, elapsed)
            run.log_metrics({f"val_macro_f1_C_{c:g}": macro_f1})
            if best is None or macro_f1 > best[1]:
                best = (c, macro_f1, pipeline)

        assert best is not None
        best_c, _, pipeline = best
        val_metrics = summarize(
            compute_metrics(val["label"].tolist(), pipeline.predict(val["text"]), LABELS)
        )
        run.log_params({"best_C": best_c})
        run.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        joblib.dump(pipeline, out / ARTIFACT, compress=3)
        card = ModelCard(
            name="sentimetro-baseline-tfidf-logreg",
            version=__version__,
            backend="sklearn",
            labels=[str(c) for c in pipeline.classes_],
            artifact=ARTIFACT,
            git_commit=git_commit(),
            mlflow_run_id=run.run_id,
            training={**params, "best_C": best_c, "top_features": top_features(pipeline)},
            validation_metrics=val_metrics,
        )
        card.save(out)
        run.log_artifact(out / "model_card.json")
        logger.info("Saved baseline to %s (val macro-F1 %.4f)", out, val_metrics["macro_f1"])
        return card


def load_pipeline(model_dir: Path) -> Pipeline:
    pipeline: Pipeline = joblib.load(model_dir / ARTIFACT)
    return pipeline


def predict_proba(pipeline: Pipeline, texts: pd.Series | list[str]) -> Any:
    return pipeline.predict_proba(texts)
