"""Request/response models (they also generate the OpenAPI docs at ``/docs``)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        "Llegó rapidísimo y funciona perfecto, muy recomendable.",
                        "Cumple, aunque la calidad del plástico es regular.",
                        "Se rompió a los dos días y el vendedor no responde.",
                    ]
                }
            ]
        },
    )

    texts: list[Text] = Field(min_length=1, description="Reseñas a clasificar (lote).")

    @field_validator("texts")
    @classmethod
    def _not_blank(cls, value: list[str]) -> list[str]:
        if any(not t for t in value):
            raise ValueError("texts must not contain empty strings")
        return value


class PredictionOut(BaseModel):
    label: str = Field(description="Clase con mayor probabilidad.")
    score: float = Field(ge=0, le=1, description="Probabilidad de la clase elegida.")
    scores: dict[str, float] = Field(description="Probabilidad de cada clase.")


class PredictResponse(BaseModel):
    model: str
    model_version: str
    predictions: list[PredictionOut]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    name: str
    version: str
    backend: str
    labels: list[str]
    max_length: int | None = None
    base_model: str | None = None
    created_at: str
    git_commit: str | None = None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    limits: dict[str, Any]
