"""API endpoints on the mock backend."""
import pytest
from fastapi.testclient import TestClient

import src.main
from conftest import INTENT_QUERIES, make_app
from src.backends import MockBackend


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(src.main, "routing_app", make_app(MockBackend(run_non_llm=False)))
    return TestClient(src.main.app)


@pytest.mark.parametrize("query", list(INTENT_QUERIES.values()))
def test_query_returns_routing_and_metrics(client, query):
    r = client.post("/query", json={"query": query})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"]
    assert body["model_used"] == body["routing_decision"]["selected_model"]
    m = body["metrics"]
    assert m["backend"] == "mock"
    assert m["backend_model"] == f"mock:{body['model_used']}"
    assert m["routing_time_ms"] > 0
    assert m["total_latency_ms"] >= m["routing_time_ms"]


def test_force_model(client):
    r = client.post("/query", json={"query": "hi", "force_model": "mistral-7b"})
    assert r.status_code == 200
    assert r.json()["model_used"] == "mistral-7b"


@pytest.mark.parametrize("model,status", [("llama-2-13b", 400), ("no-such-model", 400)])
def test_force_unavailable_model_rejected(client, model, status):
    r = client.post("/query", json={"query": "hi", "force_model": model})
    assert r.status_code == status


def test_performance_stats_include_metrics(client):
    for query in INTENT_QUERIES.values():
        client.post("/query", json={"query": query})
    stats = client.get("/performance-stats").json()
    assert stats["backend"] == "mock"
    assert stats["total_requests"] == len(INTENT_QUERIES)
    assert stats["avg_routing_time_ms"] > 0
    for model_stats in stats["model_performance"].values():
        assert model_stats["avg_latency"] > 0     # was always 0 before
        assert model_stats["success_rate"] == 1.0


def test_status_lists_available_and_unavailable(client):
    body = client.get("/status").json()
    assert body["backend"]["active"] == "mock"
    assert set(body["available_models"]) == {"phi-2", "tinyllama", "mistral-7b", "qwen3-1.7b", "qwen3-4b"}
    assert "llama-2-13b" in body["unavailable_models"]


@pytest.mark.parametrize("method,path,params", [
    ("get", "/", None),
    ("get", "/models", None),
    ("get", "/models/phi-2", None),
    ("post", "/classify-intent", {"query": "Write a Python function"}),
    ("get", "/routing-stats", None),
    ("post", "/test-query", {"query": "Tell me a story about a dragon"}),
])
def test_existing_endpoints_still_work(client, method, path, params):
    r = getattr(client, method)(path, params=params)
    assert r.status_code == 200, r.text


def test_unknown_model_404(client):
    assert client.get("/models/nope").status_code == 404


def test_models_endpoint_marks_non_llm(client):
    models = {m["name"]: m for m in client.get("/models").json()["models"]}
    assert models["business-logic-engine"]["is_llm"] is False
    assert models["mistral-7b"]["backend_model"] == "mistral"
