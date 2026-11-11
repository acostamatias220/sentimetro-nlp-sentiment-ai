"""Prometheus metrics.

Besides the usual RED metrics (rate, errors, duration) we export *model*
metrics: predicted label counts and the confidence distribution. A shift in
those over time (e.g. many more ``neutro`` or lower confidence) is a cheap,
label-free signal of data drift.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ServiceMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "sentimetro_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "sentimetro_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "route"],
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.inference_latency = Histogram(
            "sentimetro_inference_duration_seconds",
            "Model inference time per batch",
            registry=self.registry,
            buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        )
        self.batch_size = Histogram(
            "sentimetro_batch_size",
            "Texts per /predict request",
            registry=self.registry,
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        )
        self.text_chars = Histogram(
            "sentimetro_text_chars",
            "Characters per text",
            registry=self.registry,
            buckets=(20, 50, 100, 200, 500, 1000, 2000, 5000),
        )
        self.predictions = Counter(
            "sentimetro_predictions_total",
            "Predicted labels",
            ["label"],
            registry=self.registry,
        )
        self.confidence = Histogram(
            "sentimetro_prediction_confidence",
            "Probability of the predicted label",
            registry=self.registry,
            buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        )
        self.model_info = Gauge(
            "sentimetro_model_info",
            "Loaded model (value is always 1)",
            ["name", "version", "backend"],
            registry=self.registry,
        )
