"""
Integration tests for the FastAPI inference service.

Uses FastAPI's TestClient (backed by httpx) so no live server is required.
The test suite relies on trained artifacts in artifacts/ — run train.py first.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

WINDOW_SIZE = 24  # must match config.yaml


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers the lifespan, which loads
    # model/scaler/threshold/config into app.state before any test executes.
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _valid_payload() -> dict:
    return {"data_point": [float(v) for v in range(WINDOW_SIZE)]}


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_is_exact(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client: TestClient) -> None:
        response = client.post("/predict", json=_valid_payload())
        assert response.status_code == 200

    def test_response_contains_required_keys(self, client: TestClient) -> None:
        response = client.post("/predict", json=_valid_payload())
        body = response.json()
        for key in ("input_data", "anomaly_score", "is_anomaly", "threshold"):
            assert key in body, f"Missing key '{key}' in response"

    def test_anomaly_score_is_non_negative(self, client: TestClient) -> None:
        response = client.post("/predict", json=_valid_payload())
        assert response.json()["anomaly_score"] >= 0.0

    def test_is_anomaly_is_binary(self, client: TestClient) -> None:
        response = client.post("/predict", json=_valid_payload())
        assert response.json()["is_anomaly"] in (0, 1)

    def test_wrong_length_returns_422(self, client: TestClient) -> None:
        payload = {"data_point": [1.0, 2.0, 3.0]}  # too short
        response = client.post("/predict", json=payload)
        assert response.status_code in (400, 422), (
            f"Expected 400 or 422 for wrong-length payload, got {response.status_code}"
        )

    def test_wrong_length_never_returns_500(self, client: TestClient) -> None:
        payload = {"data_point": []}
        response = client.post("/predict", json=payload)
        assert response.status_code != 500

    def test_input_data_echoed_in_response(self, client: TestClient) -> None:
        payload = _valid_payload()
        response = client.post("/predict", json=payload)
        assert response.json()["input_data"] == payload["data_point"]


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_contains_expected_keys(self, client: TestClient) -> None:
        body = client.get("/model-info").json()
        for key in ("window_size", "hidden_dim", "num_layers", "threshold"):
            assert key in body


class TestRateLimiter:
    @pytest.fixture(autouse=True)
    def reset_limiter(self, client: TestClient) -> None:
        """Reset the in-memory rate-limit counter before this test.
        The MemoryStorage backend is a singleton attached to the app, so
        requests from earlier tests in the session accumulate unless cleared."""
        from src.api.main import limiter
        limiter._storage.reset()

    def test_rate_limit_trips_at_61_requests(self, client: TestClient) -> None:
        """The slowapi limiter is set to 60/minute. The 61st request in the
        same minute must be rejected with HTTP 429 Too Many Requests."""
        payload = _valid_payload()
        responses = [client.post("/predict", json=payload) for _ in range(60)]
        assert all(r.status_code == 200 for r in responses), (
            "First 60 requests must all succeed (within rate limit)"
        )
        throttled = client.post("/predict", json=payload)
        assert throttled.status_code == 429, (
            f"Expected 429 on request 61, got {throttled.status_code}"
        )


