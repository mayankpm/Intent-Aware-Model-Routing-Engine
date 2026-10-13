"""Offline evaluation of routing policies on collected outcomes.

Protocol
- Data: every item answered by every model (research/collect.py). Outcomes are
  fixed, so routing is evaluated by looking up the chosen model's real
  outcome and real measured latency for each query.
- Every learned policy is scored only on queries it was not trained on:
  repeated (R x 5-fold) cross-validation, stratified by task type.
- A policy is swept over lambda (quality vs. latency) to give an
  accuracy-vs-latency curve. Its Pareto envelope is compared with the
  envelope of randomly mixing single models (the upper hull of the
  single-model points), which any router must beat to be useful.
- Summary: area under the envelope over the latency range from the fastest
  model to the most accurate model (mean accuracy over that range of latency
  budgets), with a paired item bootstrap for differences.
- Probe cascade: the small model's first k tokens (log-probabilities and
  measured arrival times from collection) decide whether to keep its answer or
  escalate. Costs use measured times: probe time plus the large model's time.
- Conformal risk control: escalation thresholds calibrated on one split and
  checked on held-out splits.

Usage: python -m research.analyze --run main [--repeats 10]
Writes results/router_eval.json, results/router_frontier_{light,dark}.png and
the deployable router results/router/router.json.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.collect import MODELS, ROOT, load_items
from src.progress import Progress
from src.routing_policies import (ARTIFACT_PATH, EmbedLogReg, IRTRouter, KNNRouter, ProbeModel,
                                  RoutingData, TaskTable, choose, crc_threshold, latency_cost,
                                  normalize_rows, probe_time, save_artifacts)

RESULTS = ROOT.parent / "results"
LAMBDAS = np.concatenate([[0.0], np.geomspace(0.002, 4.0, 59)])
PROBE_K = [8, 16, 32]
ESTIMATORS = {
    "Task table (classified intent)": lambda: TaskTable("intent"),
    "Task table (true task type)": lambda: TaskTable("task"),
    "kNN (k=10)": lambda: KNNRouter(10),
    "Embedding logistic": lambda: EmbedLogReg(C=1.0),
    "IRT 1-D": lambda: IRTRouter(dims=1),
    "IRT 4-D": lambda: IRTRouter(dims=4),
}
NOT_DEPLOYABLE = {"Task table (true task type)"}   # needs the gold label


# --------------------------------------------------------------------------- loading

def load_data(run):
    items = {it["id"]: it for it in load_items()}
    meta = {m["id"]: m for m in map(json.loads, open(ROOT / "data" / "item_meta.jsonl", encoding="utf-8"))}
    recs = defaultdict(dict)
    for r in map(json.loads, open(ROOT / "runs" / run / "records.jsonl", encoding="utf-8")):
        recs[r["item_id"]][r["model"]] = r
    ids = [i for i in items if all(m in recs[i] for m in MODELS)]
    y = np.array([[float(recs[i][m]["passed"]) for m in MODELS] for i in ids])
    lat = np.array([[recs[i][m].get("wall_s") or np.nan for m in MODELS] for i in ids], dtype=float)
    lat = np.where(np.isnan(lat), np.nanmedian(lat, 0), lat)          # failed calls: model median
    probe = {m: [{"lp": recs[i][m].get("token_logprobs") or [], "top": recs[i][m].get("token_top_logprobs") or [],
                  "times": recs[i][m].get("token_times_s") or []} for i in ids] for m in MODELS}
    d = RoutingData(ids, list(MODELS), normalize_rows([meta[i]["embedding"] for i in ids]),
                    [meta[i]["intent"] for i in ids], [meta[i]["task"] for i in ids], y, lat, probe)
    errors = sum(1 for i in ids for m in MODELS if recs[i][m].get("error"))
    return d, [meta[i]["legacy_choice"] for i in ids], meta, recs, errors


def stratified_folds(tasks, k, seed):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(tasks), dtype=int)
    for t in sorted(set(tasks)):
        idx = np.array([i for i, x in enumerate(tasks) if x == t])
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


# --------------------------------------------------------------------------- curves

def outcome(d, choice, rows=None):
    rows = np.arange(len(choice)) if rows is None else rows
    return d.y[rows, choice[rows]].mean(), d.lat[rows, choice[rows]].mean()


def envelope(points):
    """Pareto envelope: sorted by latency, keep points that raise accuracy."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out, best = [], -1
    for lat, acc in pts:
        if acc > best + 1e-12:
            out.append((lat, acc))
            best = acc
    return out


def upper_hull(points):
    """Upper concave hull: what randomly mixing the given policies can reach."""
    pts = sorted(set(points))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return envelope(hull)


def auc(env, lo, hi, interpolate=False, n=400):
    """Mean accuracy over latency budgets in [lo, hi] (step function, or linear for hulls)."""
    if not env:
        return 0.0
    xs = np.linspace(lo, hi, n)
    lats = np.array([p[0] for p in env])
    accs = np.array([p[1] for p in env])
    if interpolate:
        vals = np.interp(xs, lats, accs, left=np.nan, right=accs[-1])
    else:
        idx = np.searchsorted(lats, xs, side="right") - 1
        vals = np.where(idx >= 0, accs[np.clip(idx, 0, None)], np.nan)
    first = accs[0]
    return float(np.nan_to_num(vals, nan=first if lats[0] <= lo else 0.0).mean())


def cascade_outcome(d, small, large, p_ok, tau, k):
    """Per-query (correct, latency) of the probe cascade small -> large."""
    s, L = MODELS.index(small), MODELS.index(large)
    keep = p_ok >= tau
    ok = np.where(keep, d.y[:, s], d.y[:, L])
    t_probe = np.array([probe_time(tr, k, d.lat[i, s]) for i, tr in enumerate(d.probe[small])])
    lat = np.where(keep, d.lat[:, s], t_probe + d.lat[:, L])
    return ok, lat, keep


# --------------------------------------------------------------------------- probe training

def oof_prior(train: RoutingData, model_idx, folds=5, seed=0):
    """Out-of-fold IRT probability for one model on the training set (no leakage into the probe)."""
    f = stratified_folds(train.task, folds, seed)
    out = np.empty(len(train.ids))
    for j in range(folds):
        tr, te = np.where(f != j)[0], np.where(f == j)[0]
        out[te] = IRTRouter(dims=1).fit(train.subset(tr)).predict(train.subset(te))[:, model_idx]
    return out


def fit_probes(train: RoutingData, smalls, ks, seed=0):
    probes = {}
    for s in smalls:
        si = MODELS.index(s)
        prior = oof_prior(train, si, seed=seed)
        for k in ks:
            probes[(s, k)] = ProbeModel(k=k).fit(train.probe[s], prior, train.y[:, si])
    return probes


# --------------------------------------------------------------------------- main evaluation

def evaluate(d, repeats, best_model):
    n, M = d.y.shape
    L = MODELS.index(best_model)
    smalls = [m for m in MODELS if d.lat[:, MODELS.index(m)].mean() < d.lat[:, L].mean() and m != best_model]
    P = {name: np.zeros((repeats, n, M)) for name in ESTIMATORS}
    P_ok = {(s, k): np.zeros((repeats, n)) for s in smalls for k in PROBE_K}
    progress = Progress(repeats * 5)
    for r in range(repeats):
        fold = stratified_folds(d.task, 5, seed=r)
        for j in range(5):
            tr, te = np.where(fold != j)[0], np.where(fold == j)[0]
            dtr, dte = d.subset(tr), d.subset(te)
            for name, make in ESTIMATORS.items():
                P[name][r][te] = make().fit(dtr).predict(dte)
            probes = fit_probes(dtr, smalls, PROBE_K, seed=r)
            irt_te = P["IRT 1-D"][r][te]
            for (s, k), pm in probes.items():
                P_ok[(s, k)][r][te] = pm.predict(dte.probe[s], irt_te[:, MODELS.index(s)])
            progress.step(f"cross-validation repeat {r + 1}/{repeats}, fold {j + 1}/5")
    return P, P_ok, smalls


def policy_curves(d, P, cost):
    """Mean (over repeats) accuracy and latency for every lambda."""
    curves = {}
    for name, pr in P.items():
        pts = []
        for lam in LAMBDAS:
            accs, lats, large_share = [], [], []
            for r in range(pr.shape[0]):
                c = choose(pr[r], cost, lam)
                a, l = outcome(d, c)
                accs.append(a)
                lats.append(l)
            pts.append({"lambda": float(lam), "acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                        "lat": float(np.mean(lats))})
        curves[name] = pts
    return curves


def cascade_curves(d, P_ok, best_model, irt_P, cost):
    """Two cascade families: fixed small->large, and IRT routing with probe escalation."""
    out = {}
    taus = np.linspace(0, 1, 41)
    for (s, k), pok in P_ok.items():
        pts = []
        for tau in taus:
            accs, lats = [], []
            for r in range(pok.shape[0]):
                ok, lat, _ = cascade_outcome(d, s, best_model, pok[r], tau, k)
                accs.append(ok.mean())
                lats.append(lat.mean())
            pts.append({"tau": float(tau), "acc": float(np.mean(accs)), "lat": float(np.mean(lats))})
        out[f"Cascade {s} -> {best_model} (k={k})"] = pts
    # IRT routing + probe escalation (k=16): route with lambda, escalate routed small models when unsure
    L = MODELS.index(best_model)
    pts = []
    for lam in LAMBDAS[::2]:
        for tau in taus[::2]:
            accs, lats = [], []
            for r in range(irt_P.shape[0]):
                c = choose(irt_P[r], cost, lam)
                ok = d.y[np.arange(len(c)), c].copy()
                lat = d.lat[np.arange(len(c)), c].copy()
                for s in {MODELS[x] for x in c} - {best_model}:
                    key = (s, 16)
                    if key not in P_ok:
                        continue
                    rows = np.where(c == MODELS.index(s))[0]
                    cok, clat, _ = cascade_outcome(d, s, best_model, P_ok[key][r], tau, 16)
                    ok[rows], lat[rows] = cok[rows], clat[rows]
                accs.append(ok.mean())
                lats.append(lat.mean())
            pts.append({"lambda": float(lam), "tau": float(tau), "acc": float(np.mean(accs)), "lat": float(np.mean(lats))})
    out["IRT + probe cascade"] = pts
    return out


def bootstrap_auc_vs_mix(d, P, name, cost, lo, hi, B=500, seed=0):
    """Paired item bootstrap of AUC(policy) - AUC(random mix of single models).

    Each draw resamples items, recomputes the single-model points and their
    upper hull on those items, and uses a random cross-validation repeat.
    """
    rng = np.random.default_rng(seed)
    n, M = d.y.shape
    R = P[name].shape[0]
    ch = np.stack([np.stack([choose(P[name][r], cost, lam) for lam in LAMBDAS]) for r in range(R)])
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, n, n)
        r = rng.integers(0, R)
        c = ch[r][:, rows]
        env = envelope(list(zip(d.lat[rows[None, :], c].mean(1), d.y[rows[None, :], c].mean(1))))
        hull = upper_hull([(float(d.lat[rows, m].mean()), float(d.y[rows, m].mean())) for m in range(M)])
        diffs.append(auc(env, lo, hi) - auc(hull, lo, hi, interpolate=True))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def accuracy_at_budget(env, budget):
    """Best envelope accuracy with mean latency at or under the budget (None if none)."""
    ok = [a for l, a in env if l <= budget + 1e-12]
    return max(ok) if ok else None


def bootstrap_auc_diff(d, P, a, b, cost, lo, hi, B=500, seed=0):
    """Paired item bootstrap of AUC(a) - AUC(b); each draw also picks a random CV repeat."""
    rng = np.random.default_rng(seed)
    n = len(d.ids)
    R = P[a].shape[0]
    ch = {name: np.stack([np.stack([choose(P[name][r], cost, lam) for lam in LAMBDAS]) for r in range(R)])
          for name in (a, b)}                                              # (R, L, n)
    diffs = []
    for _ in range(B):
        rows = rng.integers(0, n, n)
        r = rng.integers(0, R)
        aucs = []
        for name in (a, b):
            c = ch[name][r][:, rows]                                      # (L, n)
            accs = d.y[rows[None, :], c].mean(1)
            lats = d.lat[rows[None, :], c].mean(1)
            aucs.append(auc(envelope(list(zip(lats, accs))), lo, hi))
        diffs.append(aucs[0] - aucs[1])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def conformal_study(d, small, large, k=16, eps_list=(0.05, 0.10), splits=100, seed=0):
    """Probe trained on 50%, threshold calibrated by CRC on 25%, realized loss on the other 25%."""
    rng = np.random.default_rng(seed)
    n = len(d.ids)
    s, L = MODELS.index(small), MODELS.index(large)
    res = {eps: {"risk": [], "escalation": [], "latency_saving": []} for eps in eps_list}
    progress = Progress(splits)
    for sp in range(splits):
        perm = rng.permutation(n)
        tr, cal, te = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
        dtr = d.subset(tr)
        prior_tr = oof_prior(dtr, s, seed=sp)
        pm = ProbeModel(k=k).fit(dtr.probe[small], prior_tr, dtr.y[:, s])
        irt = IRTRouter(dims=1).fit(dtr)
        for part in ("cal", "te"):
            rows = cal if part == "cal" else te
            dd = d.subset(rows)
            p = pm.predict(dd.probe[small], irt.predict(dd)[:, s])
            if part == "cal":
                p_cal, dcal = p, dd
            else:
                p_te, dte = p, dd
        for eps in eps_list:
            tau = crc_threshold(p_cal, dcal.y[:, s], dcal.y[:, L], eps)
            ok, lat, keep = cascade_outcome(dte, small, large, p_te, tau, k)
            loss = (keep & (dte.y[:, s] == 0) & (dte.y[:, L] == 1)).mean()
            res[eps]["risk"].append(float(loss))
            res[eps]["escalation"].append(float(1 - keep.mean()))
            res[eps]["latency_saving"].append(float(1 - lat.mean() / dte.lat[:, L].mean()))
        progress.step(f"conformal split {sp + 1}/{splits} ({small} -> {large})")
    return {str(eps): {"mean_risk": float(np.mean(v["risk"])),
                       "share_splits_risk_above_eps": float(np.mean(np.array(v["risk"]) > eps)),
                       "mean_escalation": float(np.mean(v["escalation"])),
                       "mean_latency_saving": float(np.mean(v["latency_saving"]))} for eps, v in res.items()}


# --------------------------------------------------------------------------- plot

def plot(path, theme, singles, legacy_pt, oracle_pt, hull, lines, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                         axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
           "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                        axis="#383835", series=["#3987e5", "#d95926", "#199e70", "#c98500"])}[theme]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    fig.patch.set_facecolor(pal["surface"])
    ax.set_facecolor(pal["surface"])
    ax.grid(True, color=pal["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal["axis"])
    ax.tick_params(colors=pal["muted"])
    hx, hy = zip(*hull)
    ax.plot(hx, [v * 100 for v in hy], color=pal["muted"], linewidth=1.2, zorder=2,
            label="Random mix of single models")
    for (name, env), color in zip(lines.items(), pal["series"]):
        xs, ys = zip(*env)
        ax.step(xs, [v * 100 for v in ys], where="post", color=color, linewidth=2, label=name, zorder=3,
                solid_joinstyle="round", solid_capstyle="round")
        ax.plot(xs[-1], ys[-1] * 100, "o", color=color, markersize=7, markeredgecolor=pal["surface"],
                markeredgewidth=2, zorder=4)
    lo_x = min(x for x, _ in singles.values())
    for name, (x, yv) in singles.items():
        ax.plot(x, yv * 100, "o", color=pal["muted"], markersize=8, markeredgecolor=pal["surface"], markeredgewidth=2, zorder=5)
        ax.annotate(name, (x, yv * 100), xytext=(6, -3) if x > lo_x + 0.2 else (2, -15),
                    textcoords="offset points", color=pal["ink2"], fontsize=8.5)
    for label, pt, marker in (("legacy router", legacy_pt, "s"), ("oracle", oracle_pt, "D")):
        if pt is None:
            continue
        ax.plot(pt[0], pt[1] * 100, marker, color=pal["ink"], markersize=7, markeredgecolor=pal["surface"], markeredgewidth=2, zorder=5)
        ax.annotate(label, pt[:1] + (pt[1] * 100,), xytext=(6, 4), textcoords="offset points", color=pal["ink"], fontsize=8.5)
    ax.set_xlabel("Mean latency per query (s), measured on RTX 3070 Laptop", color=pal["ink2"])
    ax.set_ylabel("Accuracy on 240 held-out queries (%)", color=pal["ink2"])
    ax.set_title(title, color=pal["ink"], fontsize=11.5, loc="left")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(pal["ink"])
    fig.tight_layout()
    fig.savefig(path, facecolor=pal["surface"])
    plt.close(fig)


# --------------------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--conformal-splits", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict the model pool (default: every model in research/collect.py)")
    args = ap.parse_args()
    if args.models:
        MODELS[:] = args.models          # shared list: load_data and the policies see the same pool

    d, legacy, meta, recs, errors = load_data(args.run)
    n, M = d.y.shape
    lat_means = d.lat.mean(0)
    cost = latency_cost(lat_means)
    acc = d.y.mean(0)
    best = MODELS[int(np.argmax(acc))]
    B = MODELS.index(best)
    fastest = MODELS[int(np.argmin(lat_means))]
    lo, hi = float(lat_means.min()), float(lat_means[B])
    print(f"{n} items x {M} models; failed calls: {errors}; best single model: {best}; fastest: {fastest}")

    report = {"run": args.run, "items": n, "models": MODELS, "failed_calls": errors, "repeats": args.repeats}

    # ---- single models, by task
    tasks = sorted(set(d.task))
    report["single_models"] = {
        m: {"accuracy": float(acc[i]), "mean_latency_s": float(lat_means[i]),
            "median_latency_s": float(np.median(d.lat[:, i])),
            "by_task": {t: float(d.y[[j for j, x in enumerate(d.task) if x == t], i].mean()) for t in tasks}}
        for i, m in enumerate(MODELS)}

    # ---- oracle and legacy
    any_ok = d.y.max(1)
    cheapest_ok = np.array([min([j for j in range(M) if d.y[i, j]] or [int(np.argmin(lat_means))],
                                key=lambda j: d.lat[i, j]) for i in range(n)])
    oracle = outcome(d, cheapest_ok)
    report["oracle"] = {"accuracy": float(any_ok.mean()), "mean_latency_s": float(oracle[1])}
    leg = None
    if all(c in MODELS for c in legacy):
        leg = outcome(d, np.array([MODELS.index(c) for c in legacy]))
        report["legacy_router"] = {"accuracy": float(leg[0]), "mean_latency_s": float(leg[1]),
                                   "choices": dict(Counter(legacy))}
    report["classifier"] = {t: dict(Counter(x for x, tt in zip(d.intent, d.task) if tt == t)) for t in tasks}

    # ---- cross-validated policies
    P, P_ok, smalls = evaluate(d, args.repeats, best)
    curves = policy_curves(d, P, cost)
    ccurves = cascade_curves(d, P_ok, best, P["IRT 1-D"], cost)
    singles_pts = [(float(lat_means[i]), float(acc[i])) for i in range(M)]
    hull = upper_hull(singles_pts)
    hull_auc = auc(hull, lo, hi, interpolate=True)
    summary = {}
    for name, pts in {**curves, **ccurves}.items():
        env = envelope([(p["lat"], p["acc"]) for p in pts])
        max_acc = max(p["acc"] for p in pts)
        match = [p for p in pts if p["acc"] >= acc[B] - 1e-9]
        summary[name] = {
            "auc": auc(env, lo, hi), "auc_minus_random_mix": auc(env, lo, hi) - hull_auc,
            "max_accuracy": float(max_acc),
            "latency_at_max_accuracy": float(min(p["lat"] for p in pts if p["acc"] == max_acc)),
            "latency_to_match_best_model": float(min(p["lat"] for p in match)) if match else None,
        }
        if summary[name]["latency_to_match_best_model"] is not None:
            summary[name]["latency_saving_vs_best_model"] = 1 - summary[name]["latency_to_match_best_model"] / lat_means[B]
        if name in curves:
            summary[name]["accuracy_lambda0"] = curves[name][0]["acc"]
            summary[name]["accuracy_lambda0_sd_over_repeats"] = curves[name][0]["acc_sd"]
    report["random_mix_auc"] = hull_auc
    report["policies"] = summary
    report["curves"] = {**curves, **ccurves}

    # ---- noise: paired bootstrap of AUC differences
    deployable = [k for k in curves if k not in NOT_DEPLOYABLE]
    best_learned = max(deployable, key=lambda k: summary[k]["auc"])
    report["bootstrap_auc_diff_95ci"] = {
        f"{a} minus Task table (classified intent)": bootstrap_auc_diff(d, P, a, "Task table (classified intent)", cost, lo, hi, args.bootstrap)
        for a in deployable if a != "Task table (classified intent)"}
    report["bootstrap_auc_vs_random_mix_95ci"] = {
        a: bootstrap_auc_vs_mix(d, P, a, cost, lo, hi, args.bootstrap) for a in deployable}

    # ---- accuracy at matched latency: each single model's mean latency as the budget
    report["accuracy_at_single_model_latency"] = {
        name: {m: accuracy_at_budget(envelope([(p["lat"], p["acc"]) for p in pts]), float(lat_means[i]))
               for i, m in enumerate(MODELS)}
        for name, pts in {**curves, **ccurves}.items()}

    # ---- conformal cascade from the fastest decent small model
    cascade_small = max(smalls, key=lambda s: summary[f"Cascade {s} -> {best} (k=16)"]["auc"])
    report["conformal"] = {"small": cascade_small, "large": best, "k": 16,
                           "results": conformal_study(d, cascade_small, best, splits=args.conformal_splits)}

    # ---- IRT interpretation (descriptive: fit on all data)
    irt_all = IRTRouter(dims=1).fit(d)
    diff = irt_all.difficulty(d.X)
    words = np.array([meta[i]["prompt_words"] for i in d.ids])
    q = np.quantile(diff, [0.2, 0.4, 0.6, 0.8])
    bins = np.searchsorted(q, diff)
    lam_dep = None
    s_best = summary[best_learned]
    for p in curves[best_learned]:
        if s_best.get("latency_to_match_best_model") is not None and abs(p["lat"] - s_best["latency_to_match_best_model"]) < 1e-12:
            lam_dep = p["lambda"]
            break
    if lam_dep is None:
        lam_dep = 0.0
    ch0 = choose(P[best_learned][0], cost, lam_dep)
    center = float(irt_all.theta[:, 0].mean())     # only ability - difficulty matters; center for reading
    report["irt"] = {
        "ability_centered": {m: float(irt_all.theta[i, 0] - center) for i, m in enumerate(MODELS)},
        "mean_difficulty_by_task_centered": {t: float(diff[[j for j, x in enumerate(d.task) if x == t]].mean() - center) for t in tasks},
        "spearman_difficulty_vs_prompt_words": float(np.corrcoef(np.argsort(np.argsort(diff)), np.argsort(np.argsort(words)))[0, 1]),
        "routing_by_difficulty_quintile": {
            f"Q{b + 1}": {m: float((ch0[bins == b] == i).mean()) for i, m in enumerate(MODELS)} for b in range(5)},
        "accuracy_by_difficulty_quintile": {
            f"Q{b + 1}": {m: float(d.y[bins == b, i].mean()) for i, m in enumerate(MODELS)} for b in range(5)},
        "routing_policy": best_learned, "lambda": lam_dep,
    }
    # Routing by difficulty at the operating point whose mean latency is closest to the
    # middle of [fastest model, most accurate model], averaged over all CV repeats
    mid = (lo + hi) / 2
    p_mid = min(curves[best_learned], key=lambda p: abs(p["lat"] - mid))
    shares = np.zeros((5, M))
    for r in range(P[best_learned].shape[0]):
        c = choose(P[best_learned][r], cost, p_mid["lambda"])
        for b in range(5):
            shares[b] += [(c[bins == b] == i).mean() for i in range(M)]
    shares /= P[best_learned].shape[0]
    report["irt"]["mid_operating_point"] = {
        "policy": best_learned, "lambda": p_mid["lambda"], "mean_latency_s": p_mid["lat"], "accuracy": p_mid["acc"],
        "routing_by_difficulty_quintile": {f"Q{b + 1}": {m: float(shares[b, i]) for i, m in enumerate(MODELS)}
                                           for b in range(5)}}

    # ---- deployable router: best learned policy at its operating point, plus cascade
    oof = P_ok[(cascade_small, 16)][0]
    tau = crc_threshold(oof, d.y[:, MODELS.index(cascade_small)], d.y[:, B], 0.05)
    probes_all = fit_probes(d, [cascade_small], [16])
    est = ESTIMATORS[best_learned]().fit(d)
    art = {"trained_on": f"research/runs/{args.run}", "models": MODELS, "latency_means_s": lat_means.tolist(),
           "policy": best_learned, "lambda": lam_dep,
           "irt": IRTRouter(dims=1).fit(d).to_json(),
           "cascade": {"small": cascade_small, "large": best, "k": 16, "tau": tau, "eps": 0.05,
                       "probe": probes_all[(cascade_small, 16)].to_json()}}
    if isinstance(est, IRTRouter):
        art["estimator"] = {"type": "irt", **est.to_json()}
    elif isinstance(est, KNNRouter):
        art["estimator"] = {"type": "knn", "k": est.k, "temperature": est.t, "X": est.X.tolist(), "y": est.y.tolist()}
    elif isinstance(est, TaskTable):
        art["estimator"] = {"type": "table", **est.to_json()}
    elif isinstance(est, EmbedLogReg):
        art["estimator"] = {"type": "logreg", **est.to_json()}
    else:
        # Never deploy a different estimator under this policy's name
        raise RuntimeError(f"no serializer for {best_learned} ({type(est).__name__})")
    save_artifacts(ARTIFACT_PATH, **art)
    report["deployed"] = {"policy": art["policy"], "lambda": lam_dep, "cascade_tau": tau, "cascade_small": cascade_small}

    # ---- outputs
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "router_eval.json").write_text(json.dumps(report, indent=1))
    make_plots(report)
    print_report(report)


PLOT_LINES = ["Embedding logistic", "Task table (true task type)", "Task table (classified intent)",
              "IRT + probe cascade"]


def make_plots(report):
    """Light and dark accuracy-vs-latency plots from a router_eval.json report."""
    sm = report["single_models"]
    singles = {m: (s["mean_latency_s"], s["accuracy"]) for m, s in sm.items()}
    hull = upper_hull(list(singles.values()))
    lines = {name: envelope([(p["lat"], p["acc"]) for p in report["curves"][name]]) for name in PLOT_LINES}
    oracle_pt = (report["oracle"]["mean_latency_s"], report["oracle"]["accuracy"])
    lg = report.get("legacy_router")
    legacy_pt = (lg["mean_latency_s"], lg["accuracy"]) if lg else None
    for theme in ("light", "dark"):
        plot(RESULTS / f"router_frontier_{theme}.png", theme, singles, legacy_pt, oracle_pt, hull, lines,
             f"Accuracy vs. latency: routers vs. single models ({report['repeats']}x5-fold cross-validation)")


def print_report(r):
    print("\nSingle models")
    print("| Model | Accuracy | Mean latency | " + " | ".join(r["single_models"][r["models"][0]]["by_task"]) + " |")
    print("|---|---:|---:|" + "---:|" * len(r["single_models"][r["models"][0]]["by_task"]))
    for m, s in r["single_models"].items():
        print(f"| {m} | {s['accuracy']:.1%} | {s['mean_latency_s']:.2f} s | " +
              " | ".join(f"{v:.0%}" for v in s["by_task"].values()) + " |")
    print(f"\nOracle {r['oracle']['accuracy']:.1%} at {r['oracle']['mean_latency_s']:.2f} s")
    if "legacy_router" in r:
        lg = r["legacy_router"]
        print(f"Legacy router {lg['accuracy']:.1%} at {lg['mean_latency_s']:.2f} s {lg['choices']}")
    print(f"Random-mix AUC {r['random_mix_auc']:.4f}")
    print("\n| Policy | AUC | AUC - random mix | Max acc | Latency to match best model | Saving |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, s in sorted(r["policies"].items(), key=lambda kv: -kv[1]["auc"]):
        if name.startswith("Cascade") and "k=16" not in name:
            continue
        m = s.get("latency_to_match_best_model")
        print(f"| {name} | {s['auc']:.4f} | {s['auc_minus_random_mix']:+.4f} | {s['max_accuracy']:.1%} | "
              f"{'%.2f s' % m if m is not None else 'not reached'} | {('%.0f%%' % (100 * s['latency_saving_vs_best_model'])) if m is not None else '-'} |")
    print("\nBootstrap 95% CI of AUC difference:")
    for k, v in r["bootstrap_auc_diff_95ci"].items():
        print(f"  {k}: [{v[0]:+.4f}, {v[1]:+.4f}]")
    print("\nConformal cascade", r["conformal"]["small"], "->", r["conformal"]["large"], json.dumps(r["conformal"]["results"], indent=1))
    print("\nIRT abilities (centered)", json.dumps(r["irt"]["ability_centered"]))
    print("Mean difficulty by task (same scale)", json.dumps(r["irt"]["mean_difficulty_by_task_centered"]))
    print("Routing by difficulty quintile", json.dumps(r["irt"]["routing_by_difficulty_quintile"], indent=1))
    print("Deployed:", r["deployed"])
    print("\nBootstrap 95% CI of AUC minus random mix:")
    for k, v in r["bootstrap_auc_vs_random_mix_95ci"].items():
        print(f"  {k}: [{v[0]:+.4f}, {v[1]:+.4f}]")
    print("\nAccuracy at each single model's mean latency (budget):")
    for k, v in r["accuracy_at_single_model_latency"].items():
        if k.startswith("Cascade") and "k=16" not in k:
            continue
        print(f"  {k}: " + ", ".join(f"{m} {('%.1f%%' % (100 * a)) if a is not None else '-'}" for m, a in v.items()))
    mp = r["irt"]["mid_operating_point"]
    print(f"\nMid operating point ({mp['policy']}, lambda {mp['lambda']:.4f}): accuracy {mp['accuracy']:.1%} "
          f"at {mp['mean_latency_s']:.2f} s")
    print(json.dumps(mp["routing_by_difficulty_quintile"], indent=1))


if __name__ == "__main__":
    main()
