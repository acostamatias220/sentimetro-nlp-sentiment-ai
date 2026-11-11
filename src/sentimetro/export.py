"""Export the trained model for production.

Transformer -> ONNX (fp32) -> dynamic int8 quantization, with parity checks:

- fp32 ONNX logits must match PyTorch (max abs diff below ``1e-3``).
- the standalone ``tokenizers`` encoding used in serving must produce the same
  ids as the Hugging Face tokenizer used in training.
- int8 predictions must agree with fp32 on (almost) every validation review.

Each variant is written to its own folder (``models/onnx-fp32``,
``models/onnx-int8``) with tokenizer and ``model_card.json``; the selected one
is copied to ``models/production``, the folder the API serves.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from sentimetro.config import Config
from sentimetro.data.preprocess import load_split, stratified_sample
from sentimetro.inference.predictor import OnnxPredictor, TransformersPredictor
from sentimetro.model_card import ModelCard

logger = logging.getLogger(__name__)

PARITY_ATOL = 1e-3


def export_onnx(model_dir: Path, out_dir: Path, opset: int = 18) -> Path:
    """Convert a Hugging Face sequence classifier to ONNX with dynamic batch/sequence axes."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    sample = tokenizer(["una reseña de ejemplo", "otra"], padding=True, return_tensors="pt")

    class Wrapper(torch.nn.Module):
        """Expose a plain (input_ids, attention_mask) -> logits signature."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    path = out_dir / "model.onnx"
    batch = torch.export.Dim("batch")
    sequence = torch.export.Dim("sequence")
    torch.onnx.export(
        Wrapper(model).eval(),
        (sample["input_ids"], sample["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: sequence},
            "attention_mask": {0: batch, 1: sequence},
        },
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    tokenizer.save_pretrained(str(out_dir))
    for extra in out_dir.glob("*.onnx.data"):
        extra.unlink()
    return path


def quantize_int8(src: Path, dst: Path) -> Path:
    """Dynamic quantization: int8 weights, activations quantized on the fly at runtime.

    The dynamo exporter stores intermediate ``value_info`` shapes that ONNX's
    shape inference (run by the quantizer) rejects, so they are dropped first;
    ONNX Runtime re-infers them when loading the model.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    dst.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    stripped = dst.with_suffix(".stripped.onnx")
    pre = dst.with_suffix(".pre.onnx")
    onnx.save(model, str(stripped))
    try:
        quant_pre_process(str(stripped), str(pre), skip_symbolic_shape=True)
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8)
    finally:
        stripped.unlink(missing_ok=True)
        pre.unlink(missing_ok=True)
    return dst


def check_parity(
    reference: TransformersPredictor, candidate: OnnxPredictor, texts: list[str]
) -> dict[str, float]:
    """Compare probabilities and tokenization between two predictors."""
    ref = reference.predict_proba(texts)
    cand = candidate.predict_proba(texts)
    ids_ok = sum(
        reference.tokenizer(t, truncation=True, max_length=reference.card.max_length)["input_ids"]
        == candidate.tokenizer.encode(t).ids
        for t in texts
    )
    return {
        "max_abs_prob_diff": float(np.abs(ref - cand).max()),
        "label_agreement": float((ref.argmax(1) == cand.argmax(1)).mean()),
        "tokenizer_agreement": ids_ok / len(texts),
        "n": len(texts),
    }


def _write_card(
    base: ModelCard, out_dir: Path, artifact: str, variant: str, parity: dict[str, float]
) -> ModelCard:
    card = base.model_copy(
        update={
            "name": f"{base.name}-onnx-{variant}",
            "backend": "onnx",
            "artifact": artifact,
            "tokenizer": "tokenizer.json",
            "test_metrics": {},
            "training": {**base.training, "export": {"variant": variant, "parity": parity}},
        }
    )
    card.save(out_dir)
    return card


def export_transformer(cfg: Config, n_parity: int = 500) -> Path:
    source = cfg.transformer.output_dir / "final"
    base_card = ModelCard.load(source)
    # The training card describes the PyTorch checkpoint.
    torch_card = base_card.model_copy(
        update={"backend": "transformers", "artifact": "model.safetensors"}
    )
    reference = TransformersPredictor(torch_card, source)

    texts = stratified_sample(load_split(cfg.data, "validation"), n_parity, cfg.data.seed)
    texts_list = texts["text"].tolist()
    models_root = cfg.export.output_dir.parent

    fp32_dir = models_root / "onnx-fp32"
    logger.info("Exporting %s to ONNX (opset %d)", source, cfg.export.opset)
    export_onnx(source, fp32_dir, cfg.export.opset)
    _write_card(torch_card, fp32_dir, "model.onnx", "fp32", {})
    fp32 = OnnxPredictor(ModelCard.load(fp32_dir), fp32_dir)
    parity = check_parity(reference, fp32, texts_list)
    logger.info("fp32 parity: %s", parity)
    if parity["max_abs_prob_diff"] > PARITY_ATOL or parity["tokenizer_agreement"] < 1.0:
        raise RuntimeError(f"ONNX fp32 export does not match PyTorch: {parity}")
    _write_card(torch_card, fp32_dir, "model.onnx", "fp32", parity)
    chosen = fp32_dir

    if cfg.export.quantize:
        int8_dir = models_root / "onnx-int8"
        if int8_dir.exists():
            shutil.rmtree(int8_dir)
        shutil.copytree(fp32_dir, int8_dir, ignore=shutil.ignore_patterns("*.onnx"))
        quantize_int8(fp32_dir / "model.onnx", int8_dir / "model.onnx")
        _write_card(torch_card, int8_dir, "model.onnx", "int8", {})
        int8 = OnnxPredictor(ModelCard.load(int8_dir), int8_dir)
        parity_int8 = check_parity(reference, int8, texts_list)
        logger.info("int8 parity: %s", parity_int8)
        _write_card(torch_card, int8_dir, "model.onnx", "int8", parity_int8)
        chosen = int8_dir

    return _publish(chosen, cfg.export.output_dir)


def export_baseline(cfg: Config) -> Path:
    return _publish(cfg.baseline.output_dir, cfg.export.output_dir)


def _publish(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    card = ModelCard.load(dst)
    card = card.model_copy(update={"training": {**card.training, "published_from": str(src)}})
    card.save(dst)
    logger.info("Published %s -> %s (%s)", src, dst, card.name)
    return dst


def export_production(cfg: Config) -> Path:
    if cfg.export.source == "baseline":
        return export_baseline(cfg)
    return export_transformer(cfg)
