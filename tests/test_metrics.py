import numpy as np
import pytest

from sentimetro import LABELS
from sentimetro.evaluation.metrics import compute_metrics, summarize, top_errors
from sentimetro.evaluation.plots import plot_confusion_matrix


def test_compute_metrics_known_values() -> None:
    y_true = ["negativo", "negativo", "neutro", "positivo", "positivo", "positivo"]
    y_pred = ["negativo", "positivo", "neutro", "positivo", "positivo", "neutro"]
    m = compute_metrics(y_true, y_pred, LABELS)

    assert m["accuracy"] == pytest.approx(4 / 6)
    # F1 per class: negativo 2/3, neutro 2/3, positivo 2/3 -> macro 2/3
    assert m["per_class"]["negativo"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["neutro"]["f1"] == pytest.approx(2 / 3)
    assert m["per_class"]["positivo"]["f1"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[1, 0, 1], [0, 1, 0], [0, 1, 2]]
    # One negativo predicted as positivo -> 1 polarity error out of 6
    assert m["polar_error_rate"] == pytest.approx(1 / 6)


def test_macro_f1_penalizes_ignoring_minority_class() -> None:
    y_true = ["negativo"] * 45 + ["positivo"] * 45 + ["neutro"] * 10
    y_pred = ["negativo"] * 45 + ["positivo"] * 45 + ["positivo"] * 10
    m = compute_metrics(y_true, y_pred, LABELS)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["macro_f1"] < 0.65  # neutro F1 = 0 drags the macro average down


def test_summarize_is_flat() -> None:
    labels = ["negativo", "neutro", "positivo"]
    m = compute_metrics(labels, labels, LABELS)
    flat = summarize(m)
    assert flat["macro_f1"] == 1.0
    assert {"f1_negativo", "f1_neutro", "f1_positivo"} <= flat.keys()
    assert all(isinstance(v, float) for v in flat.values())


def test_top_errors_sorted_by_confidence() -> None:
    texts = ["a", "b", "c", "d"]
    y_true = ["negativo", "positivo", "neutro", "negativo"]
    probs = np.array(
        [
            [0.9, 0.05, 0.05],  # correct
            [0.7, 0.2, 0.1],  # wrong (pred negativo) conf 0.7
            [0.05, 0.05, 0.9],  # wrong (pred positivo) conf 0.9
            [0.4, 0.35, 0.25],  # correct
        ]
    )
    errors = top_errors(texts, y_true, probs, LABELS, n=5, stars=[1, 5, 3, 2])
    assert [e.text for e in errors] == ["c", "b"]
    assert errors[0].pred == "positivo"
    assert errors[0].true == "neutro"
    assert errors[0].stars == 3
    assert errors[0].confidence == pytest.approx(0.9)


def test_plot_confusion_matrix_writes_png(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = plot_confusion_matrix(
        [[3, 1, 0], [1, 2, 1], [0, 1, 4]], LABELS, "Test", tmp_path / "cm.png"
    )
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
