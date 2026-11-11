from pathlib import Path

import pandas as pd
import pytest

from sentimetro.config import DataConfig
from sentimetro.data.preprocess import (
    clean_split,
    map_stars,
    preprocess,
    remove_overlap,
    stratified_sample,
)


def _raw(rows: list[tuple[str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"es_{i}" for i in range(len(rows))],
            "text": [t for t, _ in rows],
            "label": [s for _, s in rows],
        }
    )


def test_map_stars_to_three_classes() -> None:
    cfg = DataConfig()
    mapped = map_stars(pd.Series([0, 1, 2, 3, 4]), cfg.star_to_label)
    assert mapped.tolist() == ["negativo", "negativo", "neutro", "positivo", "positivo"]


def test_map_stars_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unexpected star labels"):
        map_stars(pd.Series([0, 7]), DataConfig().star_to_label)


def test_clean_split_filters_dedupes_and_drops_conflicts() -> None:
    raw = _raw(
        [
            ("Excelente producto", 4),
            ("excelente   producto!", 4),  # duplicate after normalization
            ("ok", 2),  # too short (min_chars=3)
            ("Llegó tarde", 0),
            ("llegó tarde", 4),  # same text, different label -> ambiguous, both dropped
            ("Normal, cumple", 2),
        ]
    )
    out = clean_split(raw, DataConfig())
    assert out["text"].tolist() == ["Excelente producto", "Normal, cumple"]
    assert out["label"].tolist() == ["positivo", "neutro"]
    assert out["stars"].tolist() == [5, 3]


def test_remove_overlap_prevents_leakage() -> None:
    cfg = DataConfig()
    train = clean_split(_raw([("Muy bueno", 4), ("Muy malo", 0)]), cfg)
    test = clean_split(_raw([("muy bueno.", 4)]), cfg)
    assert remove_overlap(train, test)["text"].tolist() == ["Muy malo"]


def test_stratified_sample_keeps_proportions_and_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(1000)],
            "label": ["negativo"] * 400 + ["neutro"] * 200 + ["positivo"] * 400,
        }
    )
    a = stratified_sample(df, 100, seed=1)
    b = stratified_sample(df, 100, seed=1)
    assert len(a) == 100
    assert a["label"].value_counts().to_dict() == {"negativo": 40, "positivo": 40, "neutro": 20}
    pd.testing.assert_frame_equal(a, b)
    assert len(stratified_sample(df, None, seed=1)) == 1000


def test_preprocess_end_to_end(tmp_path: Path) -> None:
    cfg = DataConfig(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    cfg.raw_dir.mkdir()
    _raw(
        [
            ("Buenísimo, lo recomiendo", 4),
            ("Horrible", 0),
            ("Normal", 2),
            ("Lo mismo que en test", 3),
        ]
    ).to_parquet(cfg.raw_dir / "train.parquet")
    _raw([("Muy malo", 0), ("Normal", 2)]).to_parquet(cfg.raw_dir / "validation.parquet")
    _raw([("Lo mismo que en test", 3)]).to_parquet(cfg.raw_dir / "test.parquet")

    paths = preprocess(cfg)

    train = pd.read_parquet(paths["train"])
    # "Normal" is in validation and "Lo mismo…" in test: both removed from train.
    assert train["text"].tolist() == ["Buenísimo, lo recomiendo", "Horrible"]
    assert (cfg.processed_dir / "stats.json").exists()


def test_preprocess_requires_raw_data(tmp_path: Path) -> None:
    cfg = DataConfig(raw_dir=tmp_path / "missing", processed_dir=tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="make data"):
        preprocess(cfg)
