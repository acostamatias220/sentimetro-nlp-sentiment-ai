"""Fine-tune a Spanish transformer (DistilBETO by default) with HF ``Trainer``.

CPU-feasible by default (30k-review stratified subsample, ``max_length=128``,
2 epochs, dynamic padding + length-grouped batches); the same code trains on
the full dataset on a GPU by changing ``configs/params.yaml``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from sentimetro import LABELS, __version__
from sentimetro.config import Config
from sentimetro.data.preprocess import LABEL2ID, load_split, stratified_sample
from sentimetro.evaluation.metrics import compute_metrics, summarize
from sentimetro.model_card import ModelCard
from sentimetro.tracking import Run, start_run
from sentimetro.utils import environment_info, git_commit, set_seed

logger = logging.getLogger(__name__)

ID2LABEL = dict(enumerate(LABELS))


def _metrics_fn(eval_pred: Any) -> dict[str, float]:
    logits, label_ids = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    y_true = [ID2LABEL[int(i)] for i in label_ids]
    y_pred = [ID2LABEL[int(i)] for i in preds]
    return summarize(compute_metrics(y_true, y_pred, LABELS))


def class_weights(labels: list[int], n_classes: int) -> list[float]:
    """Inverse-frequency weights normalized so a balanced dataset gives all 1.0."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    weights = len(labels) / (n_classes * np.clip(counts, 1, None))
    return [float(w) for w in weights]


def _weighted_loss(weights: list[float], label_smoothing: float) -> Any:
    import torch
    import torch.nn.functional as F  # noqa: N812

    weight = torch.tensor(weights, dtype=torch.float32)

    def compute_loss(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        return F.cross_entropy(
            logits, labels, weight=weight.to(logits.device), label_smoothing=label_smoothing
        )

    return compute_loss


def _mlflow_callback(run: Run) -> Any:
    from transformers import TrainerCallback

    class MlflowLogger(TrainerCallback):
        """Forward Trainer logs (loss, lr, eval metrics) to our MLflow run."""

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
            if not logs:
                return
            numeric = {
                k: float(v)
                for k, v in logs.items()
                if isinstance(v, int | float) and math.isfinite(float(v))
            }
            run.log_metrics(numeric, step=state.global_step)

    return MlflowLogger()


def train_transformer(cfg: Config, output_dir: Path | None = None) -> ModelCard:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    t = cfg.transformer
    set_seed(cfg.data.seed)
    if t.num_threads:
        torch.set_num_threads(t.num_threads)
    out = output_dir or t.output_dir
    final_dir = out / "final"

    train_df = stratified_sample(load_split(cfg.data, "train"), t.train_samples, cfg.data.seed)
    val_df = stratified_sample(load_split(cfg.data, "validation"), t.eval_samples, cfg.data.seed)
    logger.info("Transformer: %d train / %d validation reviews", len(train_df), len(val_df))

    tokenizer = AutoTokenizer.from_pretrained(t.model_name, revision=t.model_revision)

    def to_dataset(df: Any) -> Any:
        ds = Dataset.from_dict(
            {"text": df["text"].tolist(), "labels": [LABEL2ID[x] for x in df["label"]]}
        )

        def tokenize(batch: dict[str, list[Any]]) -> Any:
            enc = tokenizer(batch["text"], truncation=True, max_length=t.max_length)
            enc["length"] = [len(ids) for ids in enc["input_ids"]]
            return enc

        return ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)
    truncated = float(np.mean(np.array(train_ds["length"]) >= t.max_length))
    logger.info("%.1f%% of training reviews truncated at %d tokens", 100 * truncated, t.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        t.model_name,
        revision=t.model_revision,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=t.epochs,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_ratio,
        lr_scheduler_type="linear",
        per_device_train_batch_size=t.train_batch_size,
        per_device_eval_batch_size=t.eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        label_smoothing_factor=t.label_smoothing,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        train_sampling_strategy="group_by_length",
        fp16=t.fp16,
        seed=cfg.data.seed,
        data_seed=cfg.data.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    params: dict[str, Any] = t.model_dump(mode="json", exclude={"output_dir"})
    params.update(train_rows=len(train_df), eval_rows=len(val_df), truncated_ratio=truncated)
    loss_fn = None
    if t.class_weighting == "balanced":
        weights = class_weights(train_ds["labels"], len(LABELS))
        loss_fn = _weighted_loss(weights, t.label_smoothing)
        params["class_weights"] = [round(w, 3) for w in weights]
        logger.info("Class weights %s", dict(zip(LABELS, params["class_weights"], strict=True)))

    with start_run(cfg.tracking, f"transformer-{t.model_name.split('/')[-1]}") as run:
        run.set_tags(
            {"model_type": "transformer", "git_commit": git_commit(), **environment_info()}
        )
        run.log_params(params)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=_metrics_fn,
            compute_loss_func=loss_fn,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=t.early_stopping_patience),
                _mlflow_callback(run),
            ],
        )
        started = time.perf_counter()
        train_result = trainer.train()
        train_minutes = (time.perf_counter() - started) / 60
        val_raw = trainer.evaluate()
        val_metrics = {
            k.removeprefix("eval_"): float(v)
            for k, v in val_raw.items()
            if k.removeprefix("eval_")
            in {
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "polar_error_rate",
                "f1_negativo",
                "f1_neutro",
                "f1_positivo",
            }
        }
        run.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        run.log_metrics({"train_minutes": train_minutes, "train_loss": train_result.training_loss})

        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        card = ModelCard(
            name=f"sentimetro-{t.model_name.split('/')[-1]}",
            version=__version__,
            backend="transformers",  # PyTorch weights; `sentimetro export` converts them to ONNX
            labels=list(LABELS),
            artifact="model.safetensors",
            tokenizer="tokenizer.json",
            max_length=t.max_length,
            base_model=f"{t.model_name}@{t.model_revision[:7]}",
            git_commit=git_commit(),
            mlflow_run_id=run.run_id,
            training={
                **params,
                "train_minutes": round(train_minutes, 1),
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "global_steps": trainer.state.global_step,
                **environment_info(),
            },
            validation_metrics=val_metrics,
        )
        card.save(final_dir)
        run.log_artifact(final_dir / "model_card.json")
        logger.info(
            "Saved transformer to %s in %.1f min (val macro-F1 %.4f)",
            final_dir,
            train_minutes,
            val_metrics.get("macro_f1", float("nan")),
        )
        return card
