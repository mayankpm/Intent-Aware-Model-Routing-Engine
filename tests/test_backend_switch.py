"""Backend selection: ROUTER_BACKEND, fallback to mock when Ollama is down."""
import pytest
from fastapi.testclient import TestClient

import src.main
from conftest import make_app
from src.backends import MockBackend, OllamaBackend, create_backend

UNREACHABLE = "http://127.0.0.1:9"  # discard port: nothing listens there


def test_mock_requested():
    sel = create_backend("mock")
    assert sel.active == "mock" and isinstance(sel.backend, MockBackend)


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("ROUTER_BACKEND", "mock")
    assert create_backend().active == "mock"


def test_ollama_unreachable_falls_back_to_mock():
    sel = create_backend("ollama", ollama_url=UNREACHABLE)
    assert sel.requested == "ollama"
    assert sel.active == "mock"
    assert isinstance(sel.backend, MockBackend)
    assert "not reachable" in sel.reason


def test_ollama_reachable_is_used(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "is_reachable", lambda self, timeout=2.0: True)
    monkeypatch.setattr(OllamaBackend, "pulled_tags", lambda self, refresh=False: {"phi:latest"})
    sel = create_backend("ollama", ollama_url=UNREACHABLE)
    assert sel.active == "ollama" and isinstance(sel.backend, OllamaBackend)


def test_invalid_backend_rejected():
    with pytest.raises(ValueError):
        create_backend("vllm")


def test_ollama_backend_maps_only_pulled_tags(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "pulled_tags",
                        lambda self, refresh=False: {"phi:latest", "tinyllama:latest", "mistral:latest"})
    app = make_app()
    backend = OllamaBackend(base_url=UNREACHABLE)
    registry = app.model_registry
    runnable = {name for name, cfg in registry.models.items() if backend.can_run(cfg)}
    assert runnable == {"phi-2", "tinyllama", "mistral-7b"}
    assert "non-LLM" in backend.unavailable_reason(registry.models["business-logic-engine"])


def test_status_reports_fallback_to_mock(monkeypatch):
    app = make_app()
    app.backend_selection = create_backend("ollama", ollama_url=UNREACHABLE)
    monkeypatch.setattr(src.main, "routing_app", app)
    body = TestClient(src.main.app).get("/status").json()
    assert body["backend"]["requested"] == "ollama"
    assert body["backend"]["active"] == "mock"
    assert "MOCK" in body["backend"]["warning"]
    assert "not reachable" in body["backend"]["reason"]
