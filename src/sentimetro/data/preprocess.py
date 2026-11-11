"""Turn the raw star-rated reviews into clean 3-class splits.

Steps (all deterministic given the seed):

1. Map stars to ``negativo`` / ``neutro`` / ``positivo``.
2. Clean the text with :func:`sentimetro.text.clean_text` and drop too-short/long reviews.
3. Remove exact duplicates inside each split (same cleaned, lowercased text).
4. Remove from train/validation any review that also appears in test
   (and from train anything that appears in validation) to avoid leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sentimetro import LABELS
from sentimetro.config import DataConfig
from sentimetro.text import clean_text

logger = logging.getLogger(__name__)

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}


def map_stars(stars: pd.Series, star_to_label: dict[int, str]) -> pd.Series:
    """Map the 0..4 star labels to class names; raises on unexpected values."""
    unknown = set(stars.unique()) - set(star_to_label)
    if unknown:
        raise ValueError(f"unexpected star labels: {sorted(unknown)}")
    return stars.map(star_to_label)


def _dedup_key(texts: pd.Series) -> pd.Series:
    return texts.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()


def clean_split(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Clean one split: map labels, normalize text, filter by length and dedupe."""
    out = pd.DataFrame(
        {
            "id": df["id"].astype(str),
            "text": df["text"].astype(str).map(clean_text),
            "stars": df["label"].astype(int) + 1,
            "label": map_stars(df["label"].astype(int), cfg.star_to_label),
        }
    )
    lengths = out["text"].str.len()
    out = out[(lengths >= cfg.min_chars) & (lengths <= cfg.max_chars)]
    out = out.assign(_key=_dedup_key(out["text"]))
    # A text duplicated with different labels is ambiguous: drop all copies.
    n_labels = out.groupby("_key")["label"].transform("nunique")
    out = out[n_labels == 1].drop_duplicates("_key")
    return out.reset_index(drop=True)


def remove_overlap(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Drop rows of ``df`` whose text also appears in ``reference``."""
    mask = df["_key"].isin(set(reference["_key"]))
    return df[~mask].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Return ``n`` rows keeping the label proportions (all rows if ``n`` is None)."""
    if n is None or n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fractions = df["label"].value_counts(normalize=True)
    quotas = (fractions * n).round().astype(int)
    # Fix rounding so the total is exactly n.
    quotas.iloc[0] += n - int(quotas.sum())
    parts = [
        df[df["label"] == label].sample(n=int(q), random_state=seed)
        for label, q in quotas.items()
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def describe(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Small statistics summary written next to the processed data."""
    stats: dict[str, Any] = {}
    for name, df in splits.items():
        words = df["text"].str.split().str.len()
        stats[name] = {
            "rows": len(df),
            "labels": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "words_median": float(words.median()),
            "words_p95": float(words.quantile(0.95)),
        }
    return stats


def preprocess(cfg: DataConfig) -> dict[str, Path]:
    """Read ``<raw_dir>/*.parquet`` and write cleaned splits to ``<processed_dir>``."""
    raw = {}
    for split in ("train", "validation", "test"):
        path = cfg.raw_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found, run `make data` (download) first")
        raw[split] = pd.read_parquet(path)

    clean = {split: clean_split(df, cfg) for split, df in raw.items()}
    clean["validation"] = remove_overlap(clean["validation"], clean["test"])
    clean["train"] = remove_overlap(clean["train"], clean["test"])
    clean["train"] = remove_overlap(clean["train"], clean["validation"])

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, df in clean.items():
        before = len(raw[split])
        df = df.drop(columns="_key")
        clean[split] = df
        path = cfg.processed_dir / f"{split}.parquet"
        df.to_parquet(path, index=False)
        paths[split] = path
        logger.info("  %-10s %7d -> %7d rows", split, before, len(df))

    stats = describe(clean)
    (cfg.processed_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def load_split(cfg: DataConfig, split: str) -> pd.DataFrame:
    path = cfg.processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `make data` first")
    return pd.read_parquet(path)
