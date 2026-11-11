from pathlib import Path

import pandas as pd
import pytest

from sentimetro.config import load_config
from sentimetro.training.baseline import build_pipeline, top_features, train_baseline
from sentimetro.training.transformer import class_weights

from .conftest import TOY_DATA


def test_class_weights_inverse_frequency() -> None:
    weights = class_weights([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], 3)
    assert weights == pytest.approx([10 / 12, 10 / 6, 10 / 12])
    assert class_weights([0, 1, 2], 3) == pytest.approx([1.0, 1.0, 1.0])


def test_baseline_pipeline_learns_toy_data() -> None:
    cfg = load_config(
        None,
        ["baseline.min_df=1", "baseline.max_features_word=1000", "baseline.max_features_char=1000"],
    )
    texts, labels = zip(*TOY_DATA, strict=True)
    pipeline = build_pipeline(cfg, c=10.0).fit(list(texts), list(labels))
    assert list(pipeline.predict(["excelente, me encantó", "horrible, no funciona"])) == [
        "positivo",
        "negativo",
    ]
    features = top_features(pipeline, k=5)
    assert set(features) == {"negativo", "neutro", "positivo"}
    assert all(len(v) == 5 for v in features.values())


def test_train_baseline_writes_model_and_card(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    rows = [(t, lab) for t, lab in TOY_DATA] * 3
    df = pd.DataFrame(
        {
            "id": range(len(rows)),
            "text": [t for t, _ in rows],
            "stars": 3,
            "label": [lab for _, lab in rows],
        }
    )
    df.to_parquet(processed / "train.parquet")
    df.to_parquet(processed / "validation.parquet")
    cfg = load_config(
        None,
        [
            f"data.processed_dir={processed}",
            f"baseline.output_dir={tmp_path / 'model'}",
            "baseline.min_df=1",
            "baseline.max_features_word=1000",
            "baseline.max_features_char=1000",
            "baseline.c_grid=[1.0, 10.0]",
            "tracking.enabled=false",
        ],
    )
    card = train_baseline(cfg)
    assert card.backend == "sklearn"
    assert card.training["best_C"] in {1.0, 10.0}
    assert card.validation_metrics["macro_f1"] > 0.9
    assert (tmp_path / "model" / "pipeline.joblib").exists()
    assert (tmp_path / "model" / "model_card.json").exists()
