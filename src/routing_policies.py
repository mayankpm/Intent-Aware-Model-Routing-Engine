"""Learned routing policies.

Every policy estimates, for a query, the probability that each model answers
it correctly. The router then picks

    argmax_m  p_m(query) - lam * cost_m

where cost_m is the model's mean measured latency divided by the slowest
model's. lam trades quality for speed; lam = 0 means "most likely correct".

Quality estimators (each from a line of LLM-routing work):

- GlobalPrior     every query gets each model's overall accuracy (no routing signal)
- TaskTable       accuracy per task type / classified intent ("Most of the LLM Routing
                  Gap Is Task Type", 2026)
- KNNRouter       mean outcome of the k nearest training queries in embedding space
                  ("When Simple kNN Beats Complex Learned Routers", 2025)
- EmbedLogReg     one logistic classifier per model on the query embedding
                  (the learned-classifier family, e.g. RouteLLM's classifiers, 2024)
- IRTRouter       item response theory: each query gets a difficulty b_q and a
                  discrimination a_q from its embedding, each model an ability theta_m;
                  P(correct) = sigmoid(a_q . theta_m - b_q) (IRT-Router, ACL 2025)

On top of a policy, ProbeCascade lets a small model start answering, reads the
log-probabilities of its first k tokens, and escalates to a large model when a
calibrated probe says the small model is likely wrong. This uses the partial
output trajectory as a routing signal ("The Routing Plateau", 2026). The
escalation threshold can be calibrated with conformal risk control
(`crc_threshold`).

Everything is numpy, deterministic given a seed, and serializable to JSON so the
API can load a trained router.
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def softplus(x):
    return np.logaddexp(0.0, x)


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


# --------------------------------------------------------------------------- data

@dataclass
class RoutingData:
    """Outcomes of every model on every query.

    y[i, m] = 1 if model m answered query i correctly; lat[i, m] = its latency (s).
    """
    ids: List[str]
    models: List[str]
    X: np.ndarray                 # (n, d) query embeddings, L2-normalized
    intent: List[str]             # intent assigned by the production classifier
    task: List[str]               # gold task type (only for the oracle task table)
    y: np.ndarray                 # (n, M) 0/1
    lat: np.ndarray               # (n, M) seconds
    probe: Dict[str, list] = field(default_factory=dict)  # model -> per-query token log-prob traces

    def subset(self, idx):
        idx = np.asarray(idx)
        return RoutingData([self.ids[i] for i in idx], self.models, self.X[idx],
                           [self.intent[i] for i in idx], [self.task[i] for i in idx],
                           self.y[idx], self.lat[idx],
                           {m: [v[i] for i in idx] for m, v in self.probe.items()})


# --------------------------------------------------------------------------- estimators

class QualityEstimator:
    name = "base"

    def fit(self, d: RoutingData):
        raise NotImplementedError

    def predict(self, d: RoutingData) -> np.ndarray:
        """(n, M) probability that each model answers each query correctly."""
        raise NotImplementedError


class GlobalPrior(QualityEstimator):
    name = "global"

    def fit(self, d):
        self.acc = (d.y.sum(0) + 1) / (len(d.y) + 2)
        return self

    def predict(self, d):
        return np.tile(self.acc, (len(d.y), 1))


class TaskTable(QualityEstimator):
    """Per-group accuracy with Laplace smoothing, backing off to global accuracy."""

    def __init__(self, key="intent", prior_strength=2.0):
        self.key, self.k = key, prior_strength
        self.name = f"table-{key}"

    def fit(self, d):
        g = GlobalPrior().fit(d).acc
        groups = getattr(d, self.key)
        self.table = {}
        for grp in set(groups):
            rows = [i for i, x in enumerate(groups) if x == grp]
            self.table[grp] = (d.y[rows].sum(0) + self.k * g) / (len(rows) + self.k)
        self.default = g
        return self

    def predict(self, d):
        return np.array([self.table.get(x, self.default) for x in getattr(d, self.key)])

    def to_json(self):
        return {"key": self.key, "table": {k: v.tolist() for k, v in self.table.items()},
                "default": self.default.tolist()}


class KNNRouter(QualityEstimator):
    """Similarity-weighted mean outcome of the k nearest training queries."""

    def __init__(self, k=10, temperature=0.05):
        self.k, self.t = k, temperature
        self.name = f"knn-{k}"

    def fit(self, d):
        self.X, self.y = d.X, d.y
        self.prior = GlobalPrior().fit(d).acc
        return self

    def predict(self, d):
        sims = d.X @ self.X.T
        k = min(self.k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = np.empty((len(d.X), self.y.shape[1]))
        for i in range(len(d.X)):
            s = sims[i, idx[i]]
            w = np.exp((s - s.max()) / self.t)
            # one pseudo-observation at the global prior keeps estimates off 0 and 1
            out[i] = (w @ self.y[idx[i]] + self.prior) / (w.sum() + 1.0)
        return out


class EmbedLogReg(QualityEstimator):
    """One L2-regularized logistic regression per model on the embedding."""

    name = "logreg"

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, d):
        from sklearn.linear_model import LogisticRegression
        self.clfs = []
        for m in range(d.y.shape[1]):
            yy = d.y[:, m]
            if yy.min() == yy.max():
                self.clfs.append(float((yy.sum() + 1) / (len(yy) + 2)))
            else:
                self.clfs.append(LogisticRegression(C=self.C, max_iter=2000).fit(d.X, yy))
        return self

    def predict(self, d):
        cols = [np.full(len(d.X), c) if isinstance(c, float) else c.predict_proba(d.X)[:, 1] for c in self.clfs]
        return np.stack(cols, 1)


class IRTRouter(QualityEstimator):
    """(Multidimensional) item response theory with embedding-predicted item parameters.

    logit P(model m correct on query q) = a_q . theta_m - b_q
      b_q = x_q . w_b + c_b                     difficulty
      a_q = softplus(x_q W_a + c_a)  (dims D)   discrimination (>= 0)
      theta_m                        (dims D)   model ability
    D = 1 is the classic, fully interpretable 2PL model.
    """

    def __init__(self, dims=1, l2=1e-2, lr=0.05, steps=600, seed=0):
        self.D, self.l2, self.lr, self.steps, self.seed = dims, l2, lr, steps, seed
        self.name = f"irt-{dims}d"

    def _forward(self, X):
        za = X @ self.Wa + self.ca
        a = softplus(za)
        b = X @ self.wb + self.cb
        return a @ self.theta.T - b[:, None], a, za, b

    def fit(self, d):
        rng = np.random.default_rng(self.seed)
        X, y = d.X, d.y
        n, dx = X.shape
        M = y.shape[1]
        acc = np.clip(y.mean(0), 0.02, 0.98)
        self.Wa = rng.normal(0, 0.01, (dx, self.D))
        self.ca = np.full(self.D, math.log(math.e - 1))          # softplus(ca) = 1
        self.wb = rng.normal(0, 0.01, dx)
        self.cb = 0.0
        self.theta = np.tile(np.log(acc / (1 - acc))[:, None], (1, self.D)) / self.D \
            + rng.normal(0, 0.01, (M, self.D))
        params = ["Wa", "ca", "wb", "cb", "theta"]
        m1 = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        m2 = {p: np.zeros_like(np.asarray(getattr(self, p), dtype=float)) for p in params}
        for t in range(1, self.steps + 1):
            _, grads = self.loss_grads(X, y)
            for p in params:                                            # Adam
                m1[p] = 0.9 * m1[p] + 0.1 * grads[p]
                m2[p] = 0.999 * m2[p] + 0.001 * grads[p] ** 2
                step = self.lr * (m1[p] / (1 - 0.9 ** t)) / (np.sqrt(m2[p] / (1 - 0.999 ** t)) + 1e-8)
                setattr(self, p, getattr(self, p) - step)
        return self

    def loss_grads(self, X, y):
        """Mean binary cross-entropy plus L2 penalties, and analytic gradients."""
        logit, a, za, b = self._forward(X)
        p = sigmoid(logit)
        reg_t = self.l2 * 1e-2
        loss = (-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
                + 0.5 * self.l2 * ((self.Wa ** 2).sum() + (self.wb ** 2).sum())
                + 0.5 * reg_t * (self.theta ** 2).sum())
        g = (p - y) / y.size                                            # dL/dlogit
        dza = (g @ self.theta) * sigmoid(za)
        db = -g.sum(1)
        return loss, {"theta": g.T @ a + reg_t * self.theta,
                      "Wa": X.T @ dza + self.l2 * self.Wa, "ca": dza.sum(0),
                      "wb": X.T @ db + self.l2 * self.wb, "cb": db.sum()}

    def predict(self, d):
        return sigmoid(self._forward(d.X)[0])

    def difficulty(self, X):
        """Interpretable query difficulty b_q (higher = harder)."""
        return X @ self.wb + self.cb

    def to_json(self):
        return {"dims": self.D, "Wa": self.Wa.tolist(), "ca": np.atleast_1d(self.ca).tolist(),
                "wb": self.wb.tolist(), "cb": float(self.cb), "theta": self.theta.tolist()}

    @classmethod
    def from_json(cls, j):
        r = cls(dims=j["dims"])
        r.Wa, r.ca, r.wb = np.array(j["Wa"]), np.array(j["ca"]), np.array(j["wb"])
        r.cb, r.theta = j["cb"], np.array(j["theta"])
        return r


# --------------------------------------------------------------------------- selection

def latency_cost(lat_means: np.ndarray) -> np.ndarray:
    return lat_means / lat_means.max()


def choose(p: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    """Index of the chosen model per query: argmax p - lam * cost (ties -> cheaper)."""
    util = p - lam * cost[None, :] - 1e-9 * cost[None, :]
    return util.argmax(1)


# --------------------------------------------------------------------------- probe cascade

def probe_features(trace: Optional[dict], k: int) -> np.ndarray:
    """Confidence features from the first k generated tokens.

    trace = {"lp": [token log-probs], "top": [[top-k log-probs] per token]}.
    Features: mean log-prob, min log-prob, share of tokens with log-prob < -1,
    mean top-1/top-2 margin, mean entropy over the renormalized top-k.
    """
    if not trace or not trace.get("lp"):
        return np.array([-5.0, -10.0, 1.0, 0.0, 1.5])
    lp = np.array(trace["lp"][:k], dtype=float)
    tops = trace.get("top") or []
    margins, ents = [], []
    for t in tops[:k]:
        if len(t) >= 2:
            s = sorted(t, reverse=True)
            margins.append(s[0] - s[1])
        if t:
            pr = np.exp(np.array(t) - max(t))
            pr /= pr.sum()
            ents.append(float(-(pr * np.log(pr + 1e-12)).sum()))
    return np.array([lp.mean(), lp.min(), (lp < -1).mean(),
                     np.mean(margins) if margins else 0.0, np.mean(ents) if ents else 0.0])


class ProbeModel:
    """Logistic model of P(small model correct | first-k-token features, router prior)."""

    def __init__(self, k=16, C=1.0):
        self.k, self.C = k, C

    def _design(self, feats, prior):
        prior = np.clip(prior, 1e-4, 1 - 1e-4)
        return np.column_stack([feats, np.log(prior / (1 - prior))])

    def fit(self, traces, prior, y):
        from sklearn.linear_model import LogisticRegression
        F = np.stack([probe_features(t, self.k) for t in traces])
        Z = self._design(F, prior)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        if y.min() == y.max():
            self.const = float((y.sum() + 1) / (len(y) + 2))
            return self
        self.const = None
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit((Z - self.mu) / self.sd, y)
        return self

    def predict(self, traces, prior):
        if self.const is not None:
            return np.full(len(traces), self.const)
        F = np.stack([probe_features(t, self.k) for t in traces])
        return self.clf.predict_proba((self._design(F, prior) - self.mu) / self.sd)[:, 1]

    def predict_one(self, lp: Sequence[float], top: Sequence[Sequence[float]], prior: float) -> float:
        return float(self.predict([{"lp": list(lp), "top": [list(t) for t in top]}], np.array([prior]))[0])

    def to_json(self):
        j = {"k": self.k, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "const": self.const}
        if self.const is None:
            j.update(coef=self.clf.coef_[0].tolist(), intercept=float(self.clf.intercept_[0]))
        return j

    @classmethod
    def from_json(cls, j):
        pm = cls(k=j["k"])
        pm.mu, pm.sd, pm.const = np.array(j["mu"]), np.array(j["sd"]), j["const"]
        if pm.const is None:
            class _Lin:  # minimal stand-in for a fitted sklearn LogisticRegression
                def __init__(self, coef, b):
                    self.coef_, self.intercept_ = np.array([coef]), np.array([b])

                def predict_proba(self, Z):
                    p = sigmoid(Z @ self.coef_[0] + self.intercept_[0])
                    return np.column_stack([1 - p, p])
            pm.clf = _Lin(j["coef"], j["intercept"])
        return pm


def probe_time(trace: Optional[dict], k: int, full_wall: float) -> float:
    """Measured time until the k-th token arrived (or the whole answer, if shorter)."""
    times = (trace or {}).get("times") or []
    if len(times) >= k:
        return float(times[k - 1])
    return float(full_wall)


def crc_threshold(p_small_ok: np.ndarray, small_ok: np.ndarray, large_ok: np.ndarray,
                  eps: float, grid: Optional[np.ndarray] = None) -> float:
    """Conformal risk control (Angelopoulos et al., 2022) for the cascade threshold.

    The cascade keeps the small model's answer when p >= tau and escalates otherwise.
    Loss per query: 1 if the cascade is wrong where the large model alone is right,
    i.e. accuracy lost against always using the large model. The loss is
    monotone non-increasing in tau. Returns the smallest tau (fewest escalations)
    whose calibration risk satisfies (n * R(tau) + 1) / (n + 1) <= eps, which
    bounds the expected loss on new exchangeable queries by eps.
    """
    n = len(p_small_ok)
    grid = np.unique(np.concatenate([[0.0, 1.0 + 1e-9], p_small_ok])) if grid is None else grid
    for tau in np.sort(grid):
        keep = p_small_ok >= tau
        loss = (keep & (small_ok == 0) & (large_ok == 1)).sum()
        if (loss + 1) / (n + 1) <= eps:
            return float(tau)
    return float(1.0 + 1e-9)   # always escalate


# --------------------------------------------------------------------------- artifacts

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "results" / "router" / "router.json"


def save_artifacts(path: Path, **parts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parts, indent=1))


def load_artifacts(path: Optional[Path] = None) -> Optional[dict]:
    path = Path(path) if path else ARTIFACT_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text())
