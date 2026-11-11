from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentimetro.inference import load_predictor
from sentimetro.serving.app import create_app
from sentimetro.serving.settings import Settings


def _client(model_dir: Path, **overrides: object) -> TestClient:
    settings = Settings(model_dir=model_dir, log_json=True, **overrides)  # type: ignore[arg-type]
    return TestClient(create_app(settings))


@pytest.fixture
def client(dummy_model_dir: Path) -> Iterator[TestClient]:
    with _client(dummy_model_dir, max_batch_size=4, max_text_chars=200) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_without_model_is_503(tmp_path: Path) -> None:
    with _client(tmp_path / "missing") as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={"texts": ["hola"]}).status_code == 503


def test_model_info(client: TestClient) -> None:
    info = client.get("/model-info").json()
    assert info["name"] == "dummy-sklearn"
    assert info["backend"] == "sklearn"
    assert info["labels"] == ["negativo", "neutro", "positivo"]
    assert info["limits"] == {"max_batch_size": 4, "max_text_chars": 200}
    assert info["training"] == {"train_rows": 12}


def test_predict_batch(client: TestClient) -> None:
    texts = ["Excelente producto, me encantó", "Horrible, se rompió", "Regular, normal"]
    response = client.post("/predict", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "dummy-sklearn"
    labels = [p["label"] for p in body["predictions"]]
    assert labels == ["positivo", "negativo", "neutro"]
    for p in body["predictions"]:
        assert set(p["scores"]) == {"negativo", "neutro", "positivo"}
        assert sum(p["scores"].values()) == pytest.approx(1.0, abs=1e-5)
        assert p["score"] == pytest.approx(max(p["scores"].values()))
    assert body["latency_ms"] >= 0


def test_predict_matches_direct_predictor(client: TestClient, dummy_model_dir: Path) -> None:
    text = "muy bueno, lo recomiendo"
    direct = load_predictor(dummy_model_dir).predict([text])[0]
    api = client.post("/predict", json={"texts": [text]}).json()["predictions"][0]
    assert api["label"] == direct.label
    assert api["score"] == pytest.approx(direct.score)


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["   "]},
        {"texts": "no es una lista"},
        {"texts": [123]},
        {"text": ["campo equivocado"]},
        {"texts": ["ok"], "extra": True},
    ],
)
def test_predict_validation_errors(client: TestClient, payload: object) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_too_many_texts(client: TestClient) -> None:
    response = client.post("/predict", json={"texts": ["bueno"] * 5})
    assert response.status_code == 422
    assert "batch too large" in response.json()["detail"]


def test_predict_rejects_too_long_texts(client: TestClient) -> None:
    response = client.post("/predict", json={"texts": ["bueno", "x" * 201]})
    assert response.status_code == 422
    assert "[1]" in response.json()["detail"]


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32


def test_prometheus_metrics(client: TestClient) -> None:
    client.post("/predict", json={"texts": ["Excelente producto", "Horrible"]})
    text = client.get("/metrics").text
    assert 'sentimetro_http_requests_total{method="POST",route="/predict",status="200"} 1.0' in text
    assert 'sentimetro_predictions_total{label="positivo"} 1.0' in text
    assert 'sentimetro_predictions_total{label="negativo"} 1.0' in text
    assert "sentimetro_batch_size_count 1.0" in text
    assert (
        'sentimetro_model_info{backend="sklearn",name="dummy-sklearn",version="0.0.1"} 1.0' in text
    )


def test_request_log_does_not_contain_review_text(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    secret_review = "mi vecino Juan Pérez dice que es excelente"
    with caplog.at_level("INFO", logger="sentimetro.access"):
        client.post("/predict", json={"texts": [secret_review]})
    records = [r for r in caplog.records if r.name == "sentimetro.access"]
    assert records
    assert records[-1].__dict__["batch_size"] == 1
    assert all(secret_review not in r.getMessage() for r in records)
    assert all(secret_review not in str(r.__dict__) for r in records)


def test_api_key_required_when_configured(dummy_model_dir: Path) -> None:
    with _client(dummy_model_dir, api_key="test-key") as c:
        payload = {"texts": ["bueno"]}
        assert c.post("/predict", json=payload).status_code == 401
        assert c.post("/predict", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
        ok = c.post("/predict", json=payload, headers={"X-API-Key": "test-key"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # ops endpoints stay open


def test_demo_page_is_served(client: TestClient) -> None:
    root = client.get("/", follow_redirects=False)
    assert root.status_code == 307
    assert root.headers["location"] == "/demo/"
    page = client.get("/demo/")
    assert page.status_code == 200
    assert "Sentímetro" in page.text
    assert client.get("/demo/app.js").status_code == 200


def test_openapi_documents_predict(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"]
    assert "/metrics" not in spec["paths"]
