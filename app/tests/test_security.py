"""API protection: shared-secret header and global rate limit, tested through the real FastAPI app."""
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import reset_rate_limiter


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "dummy")
    from app.main import app
    reset_rate_limiter()
    return TestClient(app)


def test_health_is_always_open(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "secret")
    assert client.get("/health").status_code == 200


def test_generate_rejects_missing_or_wrong_key(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "secret")
    assert client.post("/generate", json={}).status_code == 401
    assert client.post("/generate", json={}, headers={"X-API-Key": "nope"}).status_code == 401


def test_generate_with_key_reaches_validation(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "secret")
    # empty body passes auth, then fails schema validation -> 422 (not 401)
    assert client.post("/generate", json={}, headers={"X-API-Key": "secret"}).status_code == 422


def test_reports_also_protected(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "secret")
    assert client.get("/report/abc").status_code == 401


def test_open_when_no_key_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", None)
    assert client.post("/generate", json={}).status_code == 422


def test_global_rate_limit(client, monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", None)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    codes = [client.post("/generate", json={}).status_code for _ in range(5)]
    assert codes == [422, 422, 422, 429, 429]
    assert "Retry-After" in client.post("/generate", json={}).headers
