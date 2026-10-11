"""Check the offline simulation against the live system.

Runs a stratified sample of evaluation items through the deployed learned
router on Ollama (the same path as POST /query: classification, embedding,
learned selection, probe cascade with a real aborted stream) and compares with
what research/analyze.py simulated from the collected records:

- routing overhead, now including the embedding call;
- chosen model and escalation decision: live vs. simulated;
- correctness: temperature 0 and the same seed should reproduce collected answers;
- latency: live end-to-end vs. simulated (looked-up) latency.

Usage: python -m research.live_check --run main [--per-task 6]
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from research.analyze import load_data
from research.checks import score
from research.collect import MODELS, ROOT, load_items
from src.backends import create_backend
from src.main import RoutingApplication
from src.progress import Progress
from src.routing_policies import probe_time

RESULTS = ROOT.parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--per-task", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sel = create_backend("ollama")
    if sel.active != "ollama":
        raise SystemExit(sel.reason)
    app = RoutingApplication(backend_selection=sel)
    lr = app.learned_router
    if lr is None:
        raise SystemExit("no trained router: run python -m research.analyze first")
    d, _, _, recs, _ = load_data(args.run)
    items = {it["id"]: it for it in load_items()}
    rng = np.random.default_rng(args.seed)
    by_task = defaultdict(list)
    for i, t in zip(d.ids, d.task):
        by_task[t].append(i)
    sample = [i for t in sorted(by_task) for i in rng.choice(by_task[t], args.per_task, replace=False)]

    # warm the embedding model and every routed model so cold loads don't pollute the comparison
    app.backend.embed(["warm up"])
    rows = []
    progress = Progress(len(sample))
    for item_id in sample:
        it = items[item_id]
        t0 = time.perf_counter()
        decision, plan = app.routing_engine.route_with_plan(it["prompt"])
        routing_s = time.perf_counter() - t0
        options = {"num_predict": it.get("max_tokens", 256), "temperature": 0.0, "seed": 0}
        resp = app.model_manager.process_query(it["prompt"], decision.intent, decision.selected_model,
                                               options=options, routing_time=routing_s, cascade=plan)
        live_ok = bool(score(resp.response, it["check"])) if resp else False
        casc = resp.metrics.cascade if resp else None
        # simulated outcome from collected records for the same decision
        chosen = decision.selected_model
        sim_used, sim_lat = chosen, recs[item_id][chosen]["wall_s"]
        if plan is not None:
            tr = {"lp": recs[item_id][chosen].get("token_logprobs") or [],
                  "top": recs[item_id][chosen].get("token_top_logprobs") or [],
                  "times": recs[item_id][chosen].get("token_times_s") or []}
            p = plan.probe.predict_one(tr["lp"][:plan.k], tr["top"][:plan.k], plan.prior)
            if p < plan.tau:
                sim_used = plan.large
                sim_lat = probe_time(tr, plan.k, recs[item_id][chosen]["wall_s"]) + recs[item_id][plan.large]["wall_s"]
        rows.append({
            "item_id": item_id, "task": it["task"], "chosen": chosen,
            "live_model": resp.model_used if resp else None, "sim_model": sim_used,
            "escalated_live": bool(casc and casc["escalated"]), "cascade_planned": plan is not None,
            "live_ok": live_ok, "recorded_ok": bool(recs[item_id][sim_used]["passed"]),
            "same_text_as_recorded": bool(resp) and resp.response == recs[item_id][resp.model_used].get("response"),
            "routing_ms": routing_s * 1000, "live_total_ms": resp.metrics.total_latency_ms if resp else None,
            "sim_latency_ms": sim_lat * 1000,
        })
        progress.step(f"{item_id:<10} -> {rows[-1]['live_model']}{' (escalated)' if rows[-1]['escalated_live'] else ''} "
                      f"{'PASS' if live_ok else 'fail'} | routing {routing_s * 1000:.1f} ms")

    r = np.array([x["routing_ms"] for x in rows])
    summary = {
        "items": len(rows), "policy": lr.name, "lambda": lr.lam,
        "routing_ms": {"p50": float(np.median(r)), "p90": float(np.percentile(r, 90)), "max": float(r.max())},
        "model_agreement_live_vs_sim": float(np.mean([x["live_model"] == x["sim_model"] for x in rows])),
        "cascades_planned": int(sum(x["cascade_planned"] for x in rows)),
        "escalations_live": int(sum(x["escalated_live"] for x in rows)),
        "accuracy_live": float(np.mean([x["live_ok"] for x in rows])),
        "accuracy_recorded_for_same_choices": float(np.mean([x["recorded_ok"] for x in rows])),
        "identical_text_share": float(np.mean([x["same_text_as_recorded"] for x in rows])),
        "mean_latency_live_ms": float(np.mean([x["live_total_ms"] for x in rows])),
        "mean_latency_simulated_ms": float(np.mean([x["sim_latency_ms"] for x in rows])),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "router_live_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
