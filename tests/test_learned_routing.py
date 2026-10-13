"""Serving the learned router: selection, availability, probe cascade, API, fallback."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main
from conftest import make_app
from src.backends import MockBackend
from src.learned_router import LearnedRouter
from src.routing_policies import IRTRouter, ProbeModel, RoutingData, normalize_rows, save_artifacts

MODELS = ["tinyllama", "qwen3-1.7b", "phi-2", "qwen3-4b", "mistral-7b"]


def artifacts(tmp_path, lam=0.0, tau=0.5, cascade_small="qwen3-1.7b"):
    """Train a tiny IRT router on mock embeddings: abilities increase with model size."""
    backend = MockBackend()
    texts = [f"query number {i}" for i in range(120)]
    X = normalize_rows(backend.embed(texts))
    rng = np.random.default_rng(0)
    theta = np.array([-2.0, 0.5, -0.5, 1.5, 1.0])
    y = (rng.random((120, 5)) < 1 / (1 + np.exp(-theta))).astype(float)
    d = RoutingData([str(i) for i in range(120)], MODELS, X, ["x"] * 120, ["x"] * 120, y, np.ones((120, 5)))
    irt = IRTRouter(dims=1, steps=300).fit(d)
    traces, ys = [], []
    for i in range(200):
        ok = i % 2
        lp = [-0.05 if ok else -4.0] * 20
        traces.append({"lp": lp, "top": [[x, x - 1] for x in lp]})
        ys.append(ok)
    probe = ProbeModel(k=8).fit(traces, np.full(200, 0.5), np.array(ys))
    path = tmp_path / "router.json"
    save_artifacts(path, models=MODELS, latency_means_s=[0.5, 1.5, 2.5, 3.5, 7.0], policy="IRT 1-D",
                   **{"lambda": lam}, irt=irt.to_json(), estimator={"type": "irt", **irt.to_json()},
                   cascade={"small": cascade_small, "large": "qwen3-4b", "k": 8, "tau": tau, "eps": 0.05,
                            "probe": probe.to_json()})
    return path


def learned_app(tmp_path, backend=None, **kw):
    backend = backend or MockBackend(run_non_llm=False)
    app = make_app(backend)
    lr = LearnedRouter.load(backend.embed, artifacts(tmp_path, **kw))
    app.learned_router = lr
    app.routing_engine.learned_router = lr
    return app


def test_lambda_zero_picks_most_likely_correct(tmp_path):
    app = learned_app(tmp_path, lam=0.0)
    d = app.routing_engine.route_query("Explain recursion")
    assert d.policy == "learned:IRT 1-D"
    assert d.selected_model == max(d.predicted_success, key=d.predicted_success.get)
    assert d.difficulty is not None


def test_large_lambda_picks_fastest(tmp_path):
    app = learned_app(tmp_path, lam=50.0)
    assert app.routing_engine.route_query("Explain recursion").selected_model == "tinyllama"


def test_learned_router_only_picks_available_models(tmp_path):
    app = learned_app(tmp_path, backend=MockBackend(runnable=["phi-2", "tinyllama"]), lam=0.0)
    for q in ["hi", "Write a sort function", "What is 2+2?"]:
        assert app.routing_engine.route_query(q).selected_model in {"phi-2", "tinyllama"}


@pytest.mark.parametrize("logprob,escalated,used", [(-4.0, True, "qwen3-4b"), (-0.05, False, "qwen3-1.7b")])
def test_probe_cascade_escalates_only_when_unsure(tmp_path, logprob, escalated, used):
    backend = MockBackend(run_non_llm=False, token_logprob={"qwen3-1.7b": logprob})
    app = learned_app(tmp_path, backend=backend, lam=0.0, cascade_small="qwen3-1.7b")
    # Force the cascade's small model to be the pick by making the large model look slow
    app.learned_router.cost = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    app.learned_router.lam = 10.0
    from src.models import QueryRequest
    resp = app.process_query(QueryRequest(query="Explain recursion"))
    c = resp.metrics.cascade
    assert c["cascade_from"] == "qwen3-1.7b"
    assert c["escalated"] is escalated
    assert resp.model_used == used


def test_api_reports_learned_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(src.main, "routing_app", learned_app(tmp_path, lam=0.0))
    body = TestClient(src.main.app).post("/query", json={"query": "Explain recursion"}).json()
    assert body["routing_decision"]["policy"].startswith("learned:")
    assert set(body["routing_decision"]["predicted_success"]) == set(MODELS)


def test_embedding_failure_falls_back_to_legacy(tmp_path):
    app = learned_app(tmp_path)

    def boom(texts):
        raise RuntimeError("embedding model missing")
    app.learned_router.embed_fn = boom
    d = app.routing_engine.route_query("Write a Python function to sort a list")
    assert d.policy == "legacy"
    assert d.selected_model == "phi-2"


def test_logreg_artifact_is_served_as_logreg(tmp_path):
    """The deployed estimator must be the one named in the artifact (no silent substitution)."""
    from src.routing_policies import EmbedLogReg
    path = artifacts(tmp_path)
    import json
    art = json.loads(path.read_text())
    backend = MockBackend()
    X = normalize_rows(backend.embed([f"q{i}" for i in range(80)]))
    y = (np.random.default_rng(1).random((80, 5)) < 0.5).astype(float)
    d = RoutingData([str(i) for i in range(80)], MODELS, X, ["x"] * 80, ["x"] * 80, y, np.ones((80, 5)))
    est = EmbedLogReg().fit(d)
    art["policy"], art["estimator"] = "Embedding logistic", {"type": "logreg", **est.to_json()}
    path.write_text(json.dumps(art))
    lr = LearnedRouter.load(backend.embed, path)
    assert lr.kind == "logreg" and isinstance(lr.est, EmbedLogReg)
    q = RoutingData(["q"], MODELS, normalize_rows(backend.embed(["hello"])), ["x"], ["x"], np.zeros((1, 5)), np.zeros((1, 5)))
    assert np.allclose(lr.est.predict(q), est.predict(q))
