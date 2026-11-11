"""Structured (JSON) request logging with request ids.

Review texts are never logged: only counts and sizes. Reviews can contain
personal data and logs usually have a wider audience than the database.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from sentimetro.serving.metrics import ServiceMetrics

logger = logging.getLogger("sentimetro.access")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str, as_json: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if as_json
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our access log
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds ``X-Request-ID``, logs one line per request and records HTTP metrics."""

    def __init__(self, app: Any, metrics: ServiceMetrics) -> None:
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            route_path = getattr(route, "path", "unmatched")
            self.metrics.requests.labels(request.method, route_path, str(status)).inc()
            self.metrics.request_latency.labels(request.method, route_path).observe(elapsed)
            if route_path != "/metrics":
                logger.info(
                    "%s %s %d",
                    request.method,
                    request.url.path,
                    status,
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                        **getattr(request.state, "log_extra", {}),
                    },
                )
        response.headers["X-Request-ID"] = request_id
        return response
