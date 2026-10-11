"""Real inference through Ollama. Skipped when Ollama or a model is missing.

Run only these: pytest -m ollama
"""
import pytest
from fastapi.testclient import TestClient

import src.main
from conftest import INTENT_QUERIES
from src.backends import BackendSelection, OllamaBackend, normalize_tag
from src.main import RoutingApplication
from src.models import IntentType

MODELS = {"tinyllama": "tinyllama", "phi-2": "phi", "mistral-7b": "mistral"}

_backend = OllamaBackend()
_reachable = _backend.is_reachable()
_pulled = _backend.pulled_tags() if _reachable else set()

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(not _reachable, reason=f"Ollama not reachable at {_backend.base_url}"),
]


def _require(tag):
    if normalize_tag(tag) not in _pulled:
        pytest.skip(f"Ollama model '{tag}' is not pulled")


@pytest.fixture(scope="module")
def ollama_app():
    backend = OllamaBackend()
    backend.is_reachable()
    return RoutingApplication(backend_selection=BackendSelection("ollama", "ollama", "test", backend))


@pytest.mark.parametrize("name,tag", list(MODELS.items()))
def test_each_model_returns_text_and_names_itself(ollama_app, name, tag):
    _require(tag)
    resp = ollama_app.model_manager.process_query(
        "Name the capital of France in one word.", IntentType.CONVERSATION, name,
        options={"num_predict": 32, "temperature": 0, "seed": 1})
    assert resp is not None, ollama_app.model_manager.last_error
    assert resp.response.strip()
    assert resp.model_used == name
    m = resp.metrics
    assert m.backend == "ollama"
    assert normalize_tag(m.backend_model) == normalize_tag(tag)
    assert m.completion_tokens and m.completion_tokens > 0
    assert m.tokens_per_second and m.tokens_per_second > 0
    assert m.time_to_first_token_ms is not None
    # Ollama itself reports the model as resident after the call
    assert normalize_tag(tag) in {normalize_tag(t) for t in ollama_app.backend.loaded_models()}


def test_api_routes_each_intent_to_a_real_model(ollama_app, monkeypatch):
    for tag in MODELS.values():
        _require(tag)
    monkeypatch.setattr(src.main, "routing_app", ollama_app)
    client = TestClient(src.main.app)
    for intent, query in INTENT_QUERIES.items():
        r = client.post("/query", json={"query": query, "max_tokens": 24, "temperature": 0})
        assert r.status_code == 200, r.text
        body = r.json()
        info = ollama_app.model_registry.get_model_info(body["model_used"])
        assert info.is_llm and info.is_available
        assert body["metrics"]["backend"] == "ollama"
        assert normalize_tag(body["metrics"]["backend_model"]) == normalize_tag(info.backend_model)
        assert body["response"].strip()
    status = client.get("/status").json()
    assert status["backend"]["active"] == "ollama"


def test_unload_frees_model(ollama_app):
    _require("tinyllama")
    mm = ollama_app.model_manager
    assert mm._load_model("tinyllama") is not None
    assert "tinyllama" in mm.get_loaded_models()
    assert mm.unload_model("tinyllama")
    assert "tinyllama" not in mm.get_loaded_models()


def test_learned_router_on_real_models(monkeypatch):
    """With a trained router (results/router/router.json), /query routes with the learned policy."""
    from src.routing_policies import ARTIFACT_PATH
    if not ARTIFACT_PATH.exists():
        pytest.skip("no trained router; run python -m research.analyze")
    if "all-minilm:latest" not in _pulled:
        pytest.skip("embedding model all-minilm is not pulled")
    monkeypatch.setenv("ROUTER_POLICY", "learned")
    backend = OllamaBackend()
    backend.is_reachable()
    app = RoutingApplication(backend_selection=BackendSelection("ollama", "ollama", "test", backend))
    assert app.learned_router is not None
    monkeypatch.setattr(src.main, "routing_app", app)
    body = TestClient(src.main.app).post("/query", json={"query": "What is 12 times 12? Answer: <number>",
                                                         "max_tokens": 48, "temperature": 0}).json()
    assert body["routing_decision"]["policy"].startswith("learned:")
    assert body["metrics"]["backend"] == "ollama"
    assert body["response"].strip()
    info = app.model_registry.get_model_info(body["model_used"])
    assert info.is_available and info.is_llm
