from pathlib import Path

import numpy as np
import pytest

from sentimetro.inference import load_predictor
from sentimetro.inference.predictor import softmax
from sentimetro.model_card import ModelCard


def test_softmax_rows_sum_to_one() -> None:
    probs = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(probs[1], [1 / 3] * 3, rtol=1e-6)


def test_sklearn_predictor_respects_card_label_order(dummy_model_dir: Path, tmp_path: Path) -> None:
    predictor = load_predictor(dummy_model_dir)
    probs = predictor.predict_proba(["excelente producto"])
    # Same model, card declares labels in another order -> columns follow the card.
    reordered = tmp_path / "reordered"
    reordered.mkdir()
    (reordered / "pipeline.joblib").write_bytes((dummy_model_dir / "pipeline.joblib").read_bytes())
    card = ModelCard.load(dummy_model_dir)
    card.model_copy(update={"labels": ["positivo", "negativo", "neutro"]}).save(reordered)
    other = load_predictor(reordered).predict_proba(["excelente producto"])
    np.testing.assert_allclose(other[0], probs[0][[2, 0, 1]], rtol=1e-6)


def test_predictor_batches_and_cleans_text(dummy_model_dir: Path) -> None:
    predictor = load_predictor(dummy_model_dir)
    texts = ["Excelente <br/> producto"] * 5
    probs = predictor.predict_proba(texts, batch_size=2)
    assert probs.shape == (5, 3)
    clean = predictor.predict_proba(["Excelente producto"])
    np.testing.assert_allclose(probs[0], clean[0], rtol=1e-6)
    assert predictor.predict_proba([]).shape == (0, 3)


def test_missing_model_card(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictor(tmp_path)
