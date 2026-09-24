"""Security contract for public and protected API surfaces."""
from __future__ import annotations

from collections.abc import Generator

from fastapi.testclient import TestClient
import pytest

from api import main


def test_api_key_configuration_supports_legacy_render_name() -> None:
    assert main._api_key_from_environment(
        {"API_KEY": "", "VITE_LCC_API_KEY": "legacy-secret"},
    ) == "legacy-secret"


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def score_payload() -> dict[str, object]:
    return {
        "asset_type": "equity",
        "market": "US",
        "ticker": "TEST",
        "vol_ann": 0.18,
        "var95": 0.016,
        "var99": 0.025,
        "es95": 0.022,
        "es99": 0.031,
        "max_drawdown": -0.12,
        "n_used": 252,
        "missing_pct": 0.0,
    }


def test_healthz_is_public_and_minimal(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "2.0.0"}


def test_health_is_public_and_does_not_expose_internals(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "version": "2.0.0"}


def test_missing_server_api_key_fails_closed(
    client: TestClient,
    score_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "API_KEY_ENV", "")

    response = client.post("/score", json=score_payload)

    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}


def test_protected_route_rejects_missing_client_key(
    client: TestClient,
    score_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "API_KEY_ENV", "server-secret")

    response = client.post("/score", json=score_payload)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_protected_route_accepts_matching_client_key(
    client: TestClient,
    score_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "API_KEY_ENV", "server-secret")

    response = client.post(
        "/score",
        json=score_payload,
        headers={"x-api-key": "server-secret"},
    )

    assert response.status_code == 200


def test_metrics_requires_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "API_KEY_ENV", "server-secret")

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"x-api-key": "server-secret"}).status_code == 200


def test_health_details_requires_api_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "API_KEY_ENV", "server-secret")

    assert client.get("/health/details").status_code == 401
    authorized = client.get(
        "/health/details",
        headers={"x-api-key": "server-secret"},
    )
    assert authorized.status_code == 200
    assert "experts" in authorized.json()


def test_cors_does_not_allow_untrusted_browser_origins(client: TestClient) -> None:
    response = client.options(
        "/score",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") is None
