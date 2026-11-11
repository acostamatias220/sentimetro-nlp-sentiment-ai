"""Thin wrapper around MLflow so tracking can be disabled (tests, quick runs)."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from sentimetro.config import TrackingConfig

logger = logging.getLogger(__name__)


class Run:
    """Logs to MLflow when enabled, otherwise it is a no-op."""

    def __init__(self, active: bool) -> None:
        self.active = active
        self.run_id: str | None = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self.active:
            import mlflow

            mlflow.log_params({k: _short(v) for k, v in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        if self.active:
            import mlflow

            mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self.active and path.exists():
            import mlflow

            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if self.active:
            import mlflow

            mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def configure(cfg: TrackingConfig) -> None:
    import mlflow

    mlflow.set_tracking_uri(cfg.tracking_uri)
    if mlflow.get_experiment_by_name(cfg.experiment_name) is None:
        artifact_uri = Path(cfg.artifact_location).resolve().as_uri()
        mlflow.create_experiment(cfg.experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(cfg.experiment_name)


@contextlib.contextmanager
def start_run(cfg: TrackingConfig, run_name: str) -> Iterator[Run]:
    if not cfg.enabled:
        yield Run(active=False)
        return
    import mlflow

    configure(cfg)
    with mlflow.start_run(run_name=run_name) as active:
        run = Run(active=True)
        run.run_id = active.info.run_id
        logger.info("MLflow run %s (%s)", run_name, active.info.run_id)
        yield run
