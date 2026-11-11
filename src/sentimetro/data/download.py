"""Download the raw dataset from the Hugging Face Hub into ``data/raw``.

The dataset is the Spanish subset of the *Multilingual Amazon Reviews Corpus*
(Keung et al., 2020) as republished by SetFit: 200k train / 5k validation /
5k test reviews, balanced across 1-5 stars. The revision is pinned in the
config so the download is reproducible.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sentimetro.config import DataConfig

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def download_dataset(cfg: DataConfig) -> dict[str, Path]:
    """Download every split to ``<raw_dir>/<split>.parquet`` and return the paths."""
    from datasets import load_dataset  # heavy import, only needed here

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s@%s", cfg.dataset_name, cfg.dataset_revision[:7])
    dataset = load_dataset(cfg.dataset_name, revision=cfg.dataset_revision)

    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = cfg.raw_dir / f"{split}.parquet"
        dataset[split].to_parquet(str(path))
        paths[split] = path
        logger.info("  %-10s %7d rows -> %s", split, dataset[split].num_rows, path)

    manifest = {
        "dataset": cfg.dataset_name,
        "revision": cfg.dataset_revision,
        "rows": {split: dataset[split].num_rows for split in SPLITS},
    }
    (cfg.raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
