"""Export a tiny randomly initialized DistilBERT to ONNX and serve it.

Needs the ``train`` extra (torch + transformers); everything is created locally,
so nothing is downloaded from the Hub.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxscript")

from fastapi.testclient import TestClient

from sentimetro import LABELS
from sentimetro.export import check_parity, export_onnx, quantize_int8
from sentimetro.inference import load_predictor
from sentimetro.inference.predictor import OnnxPredictor, TransformersPredictor
from sentimetro.model_card import ModelCard
from sentimetro.serving.app import create_app
from sentimetro.serving.settings import Settings

pytestmark = pytest.mark.slow

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "muy",
    "bueno",
    "malo",
    "el",
    "producto",
    "una",
    "reseña",
    "de",
    "ejemplo",
    "otra",
    "no",
    "me",
    "gusto",
    "llego",
    "tarde",
    "##s",
]
TEXTS = ["Muy bueno el producto", "no me gusto, llego tarde", "una reseña de ejemplo", "otras"]


@pytest.fixture(scope="module")
def tiny_transformer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch
    from transformers import (
        BertTokenizer,
        DistilBertConfig,
        DistilBertForSequenceClassification,
    )

    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("tiny-transformer")
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    config = DistilBertConfig(
        vocab_size=len(VOCAB),
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=64,
    )
    config.id2label = dict(enumerate(LABELS))
    config.label2id = {label: i for i, label in enumerate(LABELS)}
    model_dir = directory / "final"
    DistilBertForSequenceClassification(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    ModelCard(
        name="tiny",
        version="0.0.1",
        backend="transformers",
        labels=list(LABELS),
        artifact="model.safetensors",
        tokenizer="tokenizer.json",
        max_length=32,
    ).save(model_dir)
    return model_dir


@pytest.fixture(scope="module")
def onnx_dir(tiny_transformer: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("onnx")
    export_onnx(tiny_transformer, out)
    card = ModelCard.load(tiny_transformer)
    card.model_copy(update={"backend": "onnx", "artifact": "model.onnx"}).save(out)
    return out


def test_onnx_matches_pytorch(tiny_transformer: Path, onnx_dir: Path) -> None:
    reference = load_predictor(tiny_transformer)
    candidate = load_predictor(onnx_dir)
    assert isinstance(reference, TransformersPredictor)
    assert isinstance(candidate, OnnxPredictor)
    parity = check_parity(reference, candidate, TEXTS)
    assert parity["max_abs_prob_diff"] < 1e-4
    assert parity["tokenizer_agreement"] == 1.0
    assert parity["label_agreement"] == 1.0


def test_onnx_handles_dynamic_batch_and_length(onnx_dir: Path) -> None:
    predictor = load_predictor(onnx_dir)
    probs = predictor.predict_proba(["bueno", "muy " * 100, *TEXTS])  # long text is truncated
    assert probs.shape == (6, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_int8_quantization(onnx_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "int8"
    quantize_int8(onnx_dir / "model.onnx", out / "model.onnx")
    for name in ("tokenizer.json", "tokenizer_config.json", "model_card.json"):
        (out / name).write_bytes((onnx_dir / name).read_bytes())
    fp32 = load_predictor(onnx_dir).predict_proba(TEXTS)
    int8 = load_predictor(out).predict_proba(TEXTS)
    assert np.abs(fp32 - int8).max() < 0.05
    assert not list(out.glob("*.pre.onnx"))


def test_api_serves_onnx_model(onnx_dir: Path) -> None:
    with TestClient(create_app(Settings(model_dir=onnx_dir))) as client:
        response = client.post("/predict", json={"texts": TEXTS})
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == len(TEXTS)
        assert client.get("/model-info").json()["backend"] == "onnx"
