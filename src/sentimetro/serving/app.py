"""FastAPI app: ``/predict`` (batch), ``/health``, ``/model-info``, ``/metrics`` and a demo page.

Run it with ``sentimetro serve`` or
``uvicorn sentimetro.serving.app:create_app --factory``.

No ``from __future__ import annotations`` here: FastAPI inspects the
``Annotated[..., Depends(...)]`` hints at runtime.
"""

import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentimetro import __version__
from sentimetro.inference.predictor import Predictor, load_predictor
from sentimetro.serving.logging import RequestLoggingMiddleware, configure_logging
from sentimetro.serving.metrics import ServiceMetrics
from sentimetro.serving.schemas import (
    HealthResponse,
    ModelInfo,
    PredictionOut,
    PredictRequest,
    PredictResponse,
)
from sentimetro.serving.settings import Settings

logger = logging.getLogger("sentimetro.api")

WEB_DIR = Path(__file__).resolve().parent / "web"
MODEL_INFO_TRAINING_KEYS = (
    "train_rows",
    "epochs",
    "max_length",
    "learning_rate",
    "best_C",
    "train_minutes",
    "export",
)

router = APIRouter()


# --- dependencies -----------------------------------------------------------------------


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_metrics(request: Request) -> ServiceMetrics:
    metrics: ServiceMetrics = request.app.state.metrics
    return metrics


def get_predictor(request: Request) -> Predictor:
    model: Predictor | None = request.app.state.predictor
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
    return model


def check_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    # Constant-time comparison to avoid leaking the key through timing.
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")


SettingsDep = Annotated[Settings, Depends(get_settings)]
MetricsDep = Annotated[ServiceMetrics, Depends(get_metrics)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]


# --- endpoints --------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, response: Response) -> HealthResponse:
    loaded = request.app.state.predictor is not None
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if loaded else "unavailable", model_loaded=loaded)


@router.get("/model-info", response_model=ModelInfo, tags=["model"])
def model_info(model: PredictorDep, settings: SettingsDep) -> dict[str, Any]:
    card = model.card.model_dump(mode="json")
    training = card.pop("training", {})
    return {
        **card,
        "training": {k: training[k] for k in MODEL_INFO_TRAINING_KEYS if k in training},
        "limits": {
            "max_batch_size": settings.max_batch_size,
            "max_text_chars": settings.max_text_chars,
        },
    }


@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["model"],
    dependencies=[Depends(check_api_key)],
    responses={401: {"description": "API key inválida"}, 503: {"description": "Sin modelo"}},
)
def predict(
    body: PredictRequest,
    request: Request,
    model: PredictorDep,
    settings: SettingsDep,
    metrics: MetricsDep,
) -> PredictResponse:
    # Sync endpoint on purpose: FastAPI runs it in a thread pool, so CPU-bound
    # inference never blocks the event loop.
    if len(body.texts) > settings.max_batch_size:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"batch too large: {len(body.texts)} > {settings.max_batch_size}",
        )
    too_long = [i for i, t in enumerate(body.texts) if len(t) > settings.max_text_chars]
    if too_long:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"texts {too_long} exceed {settings.max_text_chars} characters",
        )

    started = time.perf_counter()
    predictions = model.predict(body.texts)
    elapsed = time.perf_counter() - started

    metrics.inference_latency.observe(elapsed)
    metrics.batch_size.observe(len(body.texts))
    for text, pred in zip(body.texts, predictions, strict=True):
        metrics.text_chars.observe(len(text))
        metrics.predictions.labels(pred.label).inc()
        metrics.confidence.observe(pred.score)
    request.state.log_extra = {
        "batch_size": len(body.texts),
        "inference_ms": round(elapsed * 1000, 2),
    }
    return PredictResponse(
        model=model.card.name,
        model_version=model.card.version,
        predictions=[
            PredictionOut(label=p.label, score=p.score, scores=p.scores) for p in predictions
        ],
        latency_ms=round(elapsed * 1000, 2),
    )


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics(metrics: MetricsDep) -> Response:
    return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)


# --- factory ----------------------------------------------------------------------------


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    """Application factory (tests inject settings and, optionally, a predictor)."""
    settings = settings or Settings()
    metrics = ServiceMetrics()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = predictor
        if model is None:
            try:
                model = load_predictor(settings.model_dir)
            except FileNotFoundError as exc:
                logger.error("No model loaded (%s); run `make export` first", exc)
        if model is not None:
            card = model.card
            metrics.model_info.labels(card.name, card.version, card.backend).set(1)
            logger.info("Loaded model %s (%s) from %s", card.name, card.backend, model.model_dir)
        app.state.predictor = model
        yield

    app = FastAPI(
        title="Sentímetro API",
        version=__version__,
        description="Clasificación de sentimiento de reseñas en español "
        "(negativo / neutro / positivo).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.metrics = metrics
    app.state.predictor = None
    app.add_middleware(RequestLoggingMiddleware, metrics=metrics)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )
    app.include_router(router)

    if settings.enable_demo and WEB_DIR.exists():
        app.mount("/demo", StaticFiles(directory=WEB_DIR, html=True), name="demo")

        @app.get("/", include_in_schema=False)
        def root() -> RedirectResponse:
            return RedirectResponse("/demo/")

    return app
