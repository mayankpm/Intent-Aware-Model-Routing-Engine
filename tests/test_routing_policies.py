"""Learned routing policies on synthetic data with known structure."""
import numpy as np
import pytest

from src.routing_policies import (EmbedLogReg, GlobalPrior, IRTRouter, KNNRouter, ProbeModel,
                                  RoutingData, TaskTable, choose, crc_threshold, latency_cost,
                                  normalize_rows, probe_features, probe_time, sigmoid)


def synthetic(n=400, seed=0):
    """Three models with abilities -1, 0.5, 2; query difficulty is a linear function of the embedding."""
    rng = np.random.default_rng(seed)
    X = normalize_rows(rng.normal(size=(n, 16)))
    true_w = rng.normal(size=16) * 3
    b = X @ true_w
    theta = np.array([-1.0, 0.5, 2.0])
    p = sigmoid(theta[None, :] - b[:, None])
    y = (rng.random(p.shape) < p).astype(float)
    lat = np.tile([1.0, 2.0, 6.0], (n, 1))
    intent = ["easy" if bb < 0 else "hard" for bb in b]
    return RoutingData([str(i) for i in range(n)], ["s", "m", "l"], X, intent, intent, y, lat), b, theta


def test_irt_gradients_match_finite_differences():
    d, _, _ = synthetic(60)
    irt = IRTRouter(dims=2, steps=5).fit(d)
    _, grads = irt.loss_grads(d.X, d.y)
    rng = np.random.default_rng(1)
    for name in ["theta", "Wa", "ca", "wb"]:
        arr = getattr(irt, name)
        idx = tuple(rng.integers(0, s) for s in arr.shape)
        old = arr[idx]
        arr[idx] = old + 1e-5
        lp, _ = irt.loss_grads(d.X, d.y)
        arr[idx] = old - 1e-5
        lm, _ = irt.loss_grads(d.X, d.y)
        arr[idx] = old
        assert abs((lp - lm) / 2e-5 - grads[name][idx]) < 1e-6, name


def test_irt_recovers_ability_order_and_difficulty():
    d, b, theta = synthetic(600)
    irt = IRTRouter(dims=1, steps=800).fit(d)
    order = np.argsort(irt.theta[:, 0])
    assert list(order) == [0, 1, 2]
    assert np.corrcoef(irt.difficulty(d.X), b)[0, 1] > 0.8


@pytest.mark.parametrize("est", [GlobalPrior(), TaskTable("intent"), KNNRouter(10), EmbedLogReg(), IRTRouter(steps=300)])
def test_estimators_output_probabilities(est):
    d, _, _ = synthetic(200)
    p = est.fit(d.subset(range(150))).predict(d.subset(range(150, 200)))
    assert p.shape == (50, 3)
    assert np.all((p >= 0) & (p <= 1))


def test_learned_routers_beat_global_prior_on_held_out_loglik():
    d, _, _ = synthetic(800)
    tr, te = d.subset(range(600)), d.subset(range(600, 800))

    def nll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(te.y * np.log(p) + (1 - te.y) * np.log(1 - p)).mean()
    base = nll(GlobalPrior().fit(tr).predict(te))
    for est in (KNNRouter(20), EmbedLogReg(), IRTRouter(steps=600)):
        assert nll(est.fit(tr).predict(te)) < base, est.name


def test_choose_trades_quality_for_cost():
    p = np.array([[0.6, 0.7, 0.9]])
    cost = latency_cost(np.array([1.0, 2.0, 6.0]))
    assert choose(p, cost, 0.0)[0] == 2          # quality only
    assert choose(p, cost, 10.0)[0] == 0         # cost dominates
    assert choose(np.array([[0.8, 0.8, 0.8]]), cost, 0.0)[0] == 0  # ties go to the cheaper model


def test_task_table_backs_off_for_unseen_group():
    d, _, _ = synthetic(100)
    t = TaskTable("intent").fit(d)
    d2 = d.subset(range(3))
    d2.intent = ["never-seen"] * 3
    assert np.allclose(t.predict(d2), t.default)


def test_probe_features_and_time():
    trace = {"lp": [-0.1, -2.0, -0.5], "top": [[-0.1, -3.0], [-2.0, -2.1], [-0.5, -1.5]], "times": [0.1, 0.2, 0.3]}
    f = probe_features(trace, k=2)
    assert f[0] == pytest.approx(-1.05) and f[1] == -2.0 and f[2] == 0.5
    assert probe_time(trace, 2, 9.0) == 0.2
    assert probe_time(trace, 10, 9.0) == 9.0      # answer shorter than the probe: whole answer


def test_probe_model_learns_confidence_signal():
    rng = np.random.default_rng(0)
    traces, ys = [], []
    for _ in range(300):
        ok = rng.random() < 0.5
        base = -0.1 if ok else -1.5
        lp = list(base + rng.normal(0, 0.3, 16))
        traces.append({"lp": lp, "top": [[x, x - 1] for x in lp]})
        ys.append(int(ok))
    pm = ProbeModel(k=16).fit(traces[:200], np.full(200, 0.5), np.array(ys[:200]))
    p = pm.predict(traces[200:], np.full(100, 0.5))
    acc = ((p > 0.5) == np.array(ys[200:])).mean()
    assert acc > 0.9
    pm2 = ProbeModel.from_json(pm.to_json())
    assert np.allclose(pm2.predict(traces[200:], np.full(100, 0.5)), p)


def test_crc_threshold_controls_risk_on_fresh_data():
    """Average realized loss on test splits stays at or below eps (the CRC guarantee)."""
    rng = np.random.default_rng(0)
    eps, risks = 0.05, []
    for rep in range(200):
        n = 400
        p = rng.random(n)
        small_ok = (rng.random(n) < p).astype(int)        # p is a calibrated confidence
        large_ok = (rng.random(n) < 0.85).astype(int)
        cal, test = np.arange(n) < 200, np.arange(n) >= 200
        tau = crc_threshold(p[cal], small_ok[cal], large_ok[cal], eps)
        keep = p[test] >= tau
        risks.append((keep & (small_ok[test] == 0) & (large_ok[test] == 1)).mean())
    assert np.mean(risks) <= eps + 0.005


def test_irt_json_round_trip():
    d, _, _ = synthetic(100)
    irt = IRTRouter(dims=2, steps=50).fit(d)
    again = IRTRouter.from_json(irt.to_json())
    assert np.allclose(again.predict(d), irt.predict(d))


def test_embed_logreg_json_round_trip():
    d, _, _ = synthetic(150)
    lr = EmbedLogReg().fit(d)
    again = EmbedLogReg.from_json(lr.to_json())
    assert np.allclose(again.predict(d), lr.predict(d))
