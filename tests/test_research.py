"""Evaluation-set integrity, answer checkers, sandbox, and the offline analysis pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest

from research import analyze
from research.checks import (extract_choice, extract_label, extract_number, run_python_tests, score)
from src.routing_policies import RoutingData, normalize_rows, sigmoid

DATA = Path(__file__).resolve().parent.parent / "research" / "data"


def load(name):
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------- evaluation set

def test_eval_set_shape():
    counts = {n: len(load(n)) for n in ("gsm8k", "mbpp", "mmlu", "custom")}
    assert counts == {"gsm8k": 60, "mbpp": 60, "mmlu": 60, "custom": 60}
    ids = [it["id"] for n in counts for it in load(n)]
    assert len(ids) == len(set(ids))


def test_custom_references_pass_their_checks():
    assert [it["id"] for it in load("custom") if not score(it["reference"], it["check"])] == []


def test_mbpp_reference_solutions_pass_in_sandbox():
    bad = [it["id"] for it in load("mbpp") if not score(f"```python\n{it['reference']}\n```", it["check"])]
    assert bad == []


@pytest.mark.parametrize("name", ["gsm8k", "mmlu"])
def test_numeric_and_choice_answers_round_trip(name):
    for it in load(name):
        ans = it["check"]["answer"]
        assert score(f"Some reasoning.\nAnswer: {ans if name == 'mmlu' else format(ans, 'g')}", it["check"])


# ---------------------------------------------------------------- extraction

def test_extract_number_prefers_answer_marker():
    assert extract_number("We get 12 then 7.\nAnswer: 1,250") == 1250
    assert extract_number("total is $47.25") == 47.25
    assert extract_number("no digits") is None


def test_extract_label_and_choice():
    assert extract_label("It is not urgent. Answer: NORMAL", ["URGENT", "NORMAL"]) == "NORMAL"
    assert extract_label("approve? no, DENY", ["APPROVE", "DENY"]) == "DENY"
    assert extract_choice("The correct option is (C).") == "C"
    assert extract_choice("Answer: b") == "B"


def test_wrong_answers_fail():
    assert not score("Answer: 41", {"type": "number", "answer": 42})
    assert not score("line one", {"type": "constraints", "lines": 4})
    assert not score("```python\ndef f(x): return x\n```", {"type": "python", "tests": ["assert f(2) == 4"]})


# ---------------------------------------------------------------- sandbox

@pytest.mark.parametrize("code", [
    "open('x.txt', 'w').write('hi')",
    "import os; os.remove('somefile')",
    "import subprocess; subprocess.run(['cmd', '/c', 'echo'])",
    "import socket; socket.create_connection(('example.com', 80))",
    "input()",
    "while True: pass",
])
def test_sandbox_blocks_side_effects(code):
    assert not run_python_tests(code, [], timeout=3)


def test_sandbox_runs_normal_code():
    assert run_python_tests("import math\ndef f(x): return math.sqrt(x)", ["assert f(9) == 3"])


# ---------------------------------------------------------------- analysis pipeline

def fake_data(n=120, seed=0):
    """Five models of increasing ability and latency; difficulty depends on the embedding."""
    rng = np.random.default_rng(seed)
    X = normalize_rows(rng.normal(size=(n, 8)))
    b = X @ rng.normal(size=8) * 2
    theta = np.array([-1.5, 0.0, 0.5, 1.5, 1.0])
    y = (rng.random((n, 5)) < sigmoid(theta[None] - b[:, None])).astype(float)
    lat = np.tile([0.5, 1.5, 2.5, 3.5, 7.0], (n, 1)) * rng.uniform(0.8, 1.2, (n, 5))
    tasks = [["math", "code", "knowledge"][i % 3] for i in range(n)]
    probe = {}
    for j, m in enumerate(analyze.MODELS):
        probe[m] = [{"lp": list(rng.normal(-0.2 if y[i, j] else -1.2, 0.3, 40)),
                     "top": [[-0.1, -2.0]] * 40, "times": list(np.linspace(0.05, 2, 40))} for i in range(n)]
    return RoutingData([str(i) for i in range(n)], list(analyze.MODELS), X, tasks, tasks, y, lat, probe)


def test_envelope_and_hull():
    pts = [(1, 0.5), (2, 0.4), (3, 0.7), (4, 0.65), (5, 0.9)]
    assert analyze.envelope(pts) == [(1, 0.5), (3, 0.7), (5, 0.9)]
    hull = analyze.upper_hull(pts)
    assert hull[0] == (1, 0.5) and hull[-1] == (5, 0.9)
    assert analyze.auc([(1, 0.5), (5, 0.9)], 1, 5) == pytest.approx(0.5, abs=0.01)


def test_pipeline_runs_and_learned_router_beats_random_mix(monkeypatch):
    d = fake_data()
    monkeypatch.setattr(analyze, "PROBE_K", [16])
    cost = np.array([0.5, 1.5, 2.5, 3.5, 7.0]) / 7.0
    P, P_ok, smalls = analyze.evaluate(d, repeats=1, best_model="qwen3-4b")
    assert set(P) == set(analyze.ESTIMATORS)
    curves = analyze.policy_curves(d, P, cost)
    lat_means = d.lat.mean(0)
    hull = analyze.upper_hull([(float(lat_means[i]), float(d.y[:, i].mean())) for i in range(5)])
    lo, hi = float(lat_means.min()), float(lat_means[3])
    hull_auc = analyze.auc(hull, lo, hi, interpolate=True)
    irt_auc = analyze.auc(analyze.envelope([(p["lat"], p["acc"]) for p in curves["IRT 1-D"]]), lo, hi)
    assert irt_auc > hull_auc
    cc = analyze.cascade_curves(d, P_ok, "qwen3-4b", P["IRT 1-D"], cost)
    assert "IRT + probe cascade" in cc
    res = analyze.conformal_study(d, "tinyllama", "qwen3-4b", eps_list=(0.1,), splits=3)
    assert 0 <= res["0.1"]["mean_risk"] <= 1
