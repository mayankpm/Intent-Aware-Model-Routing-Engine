"""Serve the router trained by research/analyze.py.

Loads results/router/router.json (an IRT, kNN or task-table quality estimator,
the quality/latency trade-off lambda, and the probe-cascade settings), embeds
each query with the same embedding model used in training, and picks
argmax_m P(model m correct) - lambda * latency_cost_m among the models that are
currently available. When the pick is the cascade's small model, it also
returns a cascade plan: run the small model, read its first k tokens, and
escalate to the large model if the probe's P(correct) falls below tau.
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .routing_policies import (EmbedLogReg, IRTRouter, ProbeModel, RoutingData, TaskTable, KNNRouter,
                               choose, latency_cost, load_artifacts, normalize_rows)

logger = logging.getLogger(__name__)


@dataclass
class CascadePlan:
    small: str
    large: str
    k: int
    tau: float
    probe: ProbeModel
    prior: float          # router's P(small correct), the probe's prior feature


@dataclass
class LearnedChoice:
    model: str
    p_correct: Dict[str, float]
    difficulty: Optional[float]
    cascade: Optional[CascadePlan]


class LearnedRouter:
    def __init__(self, artifacts: dict, embed_fn, use_cascade: bool = True):
        self.art = artifacts
        self.embed_fn = embed_fn
        self.models: List[str] = artifacts["models"]
        self.cost = latency_cost(np.array(artifacts["latency_means_s"]))
        self.lam = float(os.getenv("ROUTER_LAMBDA", artifacts.get("lambda", 0.0)))
        est = artifacts["estimator"]
        self.kind = est["type"]
        if self.kind == "irt":
            self.est = IRTRouter.from_json(est)
        elif self.kind == "knn":
            self.est = KNNRouter(est["k"], est["temperature"])
            self.est.X, self.est.y = np.array(est["X"]), np.array(est["y"])
            self.est.prior = (self.est.y.sum(0) + 1) / (len(self.est.y) + 2)
        elif self.kind == "logreg":
            self.est = EmbedLogReg.from_json(est)
        elif self.kind == "table":
            self.est = TaskTable(est["key"])
            self.est.table = {k: np.array(v) for k, v in est["table"].items()}
            self.est.default = np.array(est["default"])
        else:
            raise ValueError(f"unknown estimator type {self.kind}")
        self.irt = IRTRouter.from_json(artifacts["irt"])     # for difficulty readout and probe prior
        c = artifacts.get("cascade")
        self.cascade = None
        if use_cascade and c:
            self.cascade = dict(small=c["small"], large=c["large"], k=int(c["k"]), tau=float(c["tau"]),
                                probe=ProbeModel.from_json(c["probe"]))
        self.name = f"learned:{artifacts['policy']}"

    @classmethod
    def load(cls, embed_fn, path=None, use_cascade=True) -> Optional["LearnedRouter"]:
        art = load_artifacts(path)
        if art is None:
            return None
        return cls(art, embed_fn, use_cascade)

    def choose(self, query: str, intent: str, available: List[str]) -> Optional[LearnedChoice]:
        candidates = [m for m in self.models if m in available]
        if not candidates:
            return None
        x = normalize_rows([self.embed_fn([query])[0]])
        d = RoutingData(["q"], self.models, x, [intent], [intent], np.zeros((1, len(self.models))),
                        np.zeros((1, len(self.models))))
        p = self.est.predict(d)[0]
        mask = np.array([m in candidates for m in self.models])
        util_p = np.where(mask, p, -np.inf)
        idx = int(choose(util_p[None, :], self.cost, self.lam)[0])
        model = self.models[idx]
        plan = None
        if self.cascade and model == self.cascade["small"] and self.cascade["large"] in candidates:
            prior = float(self.irt.predict(d)[0][self.models.index(model)])
            plan = CascadePlan(model, self.cascade["large"], self.cascade["k"], self.cascade["tau"],
                               self.cascade["probe"], prior)
        return LearnedChoice(model, {m: float(v) for m, v in zip(self.models, p)},
                             float(self.irt.difficulty(x)[0]), plan)
