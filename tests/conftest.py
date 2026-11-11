"""Shared fixtures: a tiny sklearn model trained in milliseconds (no downloads)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentimetro import LABELS
from sentimetro.model_card import ModelCard

TOY_DATA = [
    ("excelente producto, me encantó", "positivo"),
    ("muy bueno, llegó rápido y funciona perfecto", "positivo"),
    ("lo recomiendo, gran calidad", "positivo"),
    ("perfecto, cinco estrellas", "positivo"),
    ("horrible, se rompió enseguida", "negativo"),
    ("muy malo, no funciona", "negativo"),
    ("una estafa, no lo compren", "negativo"),
    ("pésima calidad, lo devolví", "negativo"),
    ("está bien, normal, cumple", "neutro"),
    ("regular, ni bueno ni malo", "neutro"),
    ("aceptable por el precio, normal", "neutro"),
    ("correcto sin más, regular", "neutro"),
]


@pytest.fixture(scope="session")
def dummy_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("dummy-model")
    texts, labels = zip(*TOY_DATA, strict=True)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(C=10, max_iter=500, random_state=0)),
        ]
    ).fit(list(texts), list(labels))
    joblib.dump(pipeline, directory / "pipeline.joblib")
    ModelCard(
        name="dummy-sklearn",
        version="0.0.1",
        backend="sklearn",
        labels=list(LABELS),
        artifact="pipeline.joblib",
        training={"train_rows": len(TOY_DATA)},
        validation_metrics={"macro_f1": 1.0},
    ).save(directory)
    return directory
