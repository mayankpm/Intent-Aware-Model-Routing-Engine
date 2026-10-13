"""Print the README results tables from results/*.json.

Every number in the README's research results comes from this script.
Rounding is conservative: accuracies and gains round down, latencies and
overheads round up, and confidence intervals round outward.

Usage: python -m research.report_tables
"""
import json
import math
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def down(x, nd=1):
    f = 10 ** nd
    return f"{math.floor(x * f + 1e-9) / f:.{nd}f}"


def up(x, nd=1):
    f = 10 ** nd
    return f"{math.ceil(x * f - 1e-9) / f:.{nd}f}"


def ci(lo, hi, nd=3):
    return f"[{'+' if lo >= 0 else ''}{down(lo, nd)}, {'+' if hi >= 0 else ''}{up(hi, nd)}]"


def signed_down(x, nd=3):
    s = down(x, nd)
    return s if s.startswith("-") else "+" + s


def main():
    ev = json.loads((R / "router_eval.json").read_text())
    sm = ev["single_models"]
    tasks = list(next(iter(sm.values()))["by_task"])
    print("### Single models\n")
    print("| Model | Correct of 240 | Accuracy | Mean latency | " + " | ".join(t.replace("_", " ") for t in tasks) + " |")
    print("|---|---:|---:|---:|" + "---:|" * len(tasks))
    for m, s in sm.items():
        n_ok = round(s["accuracy"] * ev["items"])
        print(f"| {m} | {n_ok} | {down(100 * s['accuracy'])}% | {up(s['mean_latency_s'], 2)} s | "
              + " | ".join(f"{down(100 * v)}%" for v in s["by_task"].values()) + " |")
    o = ev["oracle"]
    print(f"\nOracle (cheapest correct model per query): {down(100 * o['accuracy'])}% at {up(o['mean_latency_s'], 2)} s")
    print(f"Random-mix AUC: {down(ev['random_mix_auc'], 4)}")

    pol = ev["policies"]
    mix_ci = ev["bootstrap_auc_vs_random_mix_95ci"]
    tab_ci = ev["bootstrap_auc_diff_95ci"]
    at = ev["accuracy_at_single_model_latency"]
    names = ["Embedding logistic", "IRT 4-D", "kNN (k=10)", "IRT 1-D", "Task table (classified intent)",
             "Task table (true task type)", "IRT + probe cascade"]
    ref_small = "qwen3-1.7b"
    print("\n### Policies\n")
    print(f"| Policy | AUC | AUC minus random mix | 95% interval vs random mix | 95% interval vs intent table | "
          f"Accuracy at {ref_small} latency | Best accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for n in names:
        p = pol[n]
        a = at[n][ref_small]
        vs_tab = tab_ci.get(f"{n} minus Task table (classified intent)")
        print(f"| {n} | {down(p['auc'], 4)} | {signed_down(p['auc_minus_random_mix'], 4)} | "
              f"{ci(*mix_ci[n]) if n in mix_ci else 'n/a'} | {ci(*vs_tab) if vs_tab else 'n/a'} | "
              f"{down(100 * a) + '%' if a is not None else 'n/a'} | {down(100 * p['max_accuracy'])}% |")
    for n in [k for k in pol if k.startswith("Cascade") and "k=16" in k]:
        print(f"| {n} | {down(pol[n]['auc'], 4)} | {signed_down(pol[n]['auc_minus_random_mix'], 4)} | n/a | n/a | "
              f"{(down(100 * at[n][ref_small]) + '%') if at[n][ref_small] is not None else 'n/a'} | {down(100 * pol[n]['max_accuracy'])}% |")

    print("\n### Accuracy at each single model's latency budget\n")
    models = list(sm)
    print("| Policy | " + " | ".join(f"{m} ({up(sm[m]['mean_latency_s'], 2)} s)" for m in models) + " |")
    print("|---|" + "---:|" * len(models))
    print("| the model itself | " + " | ".join(f"{down(100 * sm[m]['accuracy'])}%" for m in models) + " |")
    for n in ["Embedding logistic", "Task table (classified intent)", "Task table (true task type)"]:
        print(f"| {n} | " + " | ".join((down(100 * at[n][m]) + "%") if at[n][m] is not None else "n/a" for m in models) + " |")

    irt = ev["irt"]
    mp = irt["mid_operating_point"]
    print(f"\n### Mid operating point: {mp['policy']}, lambda {mp['lambda']:.4f}, "
          f"accuracy {down(100 * mp['accuracy'])}% at {up(mp['mean_latency_s'], 2)} s\n")
    print("| Difficulty fifth | " + " | ".join(models) + " |")
    print("|---|" + "---:|" * len(models))
    for q, shares in mp["routing_by_difficulty_quintile"].items():
        print(f"| {q} | " + " | ".join(f"{round(100 * shares[m])}%" for m in models) + " |")
    print("\nIRT ability (centered):", {m: round(v, 2) for m, v in irt["ability_centered"].items()})
    print("IRT mean difficulty by task (same scale):", {t: round(v, 2) for t, v in irt["mean_difficulty_by_task_centered"].items()})
    print("Spearman(difficulty, prompt words):", round(irt["spearman_difficulty_vs_prompt_words"], 2))
    print("Classifier labels by task:", json.dumps(ev["classifier"]))

    c = ev["conformal"]
    print(f"\n### Conformal cascade {c['small']} -> {c['large']} (k={c['k']})\n")
    print("| Risk target | Mean realized loss | Splits above target | Mean escalation | Mean latency saving |")
    print("|---:|---:|---:|---:|---:|")
    for eps, v in c["results"].items():
        print(f"| {eps} | {up(v['mean_risk'], 3)} | {round(100 * v['share_splits_risk_above_eps'])}% | "
              f"{round(100 * v['mean_escalation'])}% | {down(100 * v['mean_latency_saving'])}% |")
    print("\nDeployed:", json.dumps(ev["deployed"]))

    print("\n### Live checks\n")
    print("| Run | lambda | Items | Routing p50 | Routing p90 | Same model as simulation | Cascades planned | "
          "Escalated live | Same correctness as recorded | Identical text | Median latency live | Median simulated |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, f in (("deployed", "router_live_check.json"), ("mid-frontier", "router_live_check_mid.json"),
                     ("cascade forced", "router_live_check_cascade.json")):
        p = R / f
        if not p.exists():
            continue
        s = json.loads(p.read_text())["summary"]
        n = s["items"]
        print(f"| {label} | {s['lambda']:.4g} | {n} | {up(s['routing_ms']['p50'])} ms | {up(s['routing_ms']['p90'])} ms | "
              f"{round(s['model_agreement_live_vs_sim'] * n)} of {n} | {s['cascades_planned']} | {s['escalations_live']} | "
              f"{s['correctness_agreement_per_item']} of {n} | {round(s['identical_text_share'] * n)} of {n} | "
              f"{up(s['median_latency_live_ms'], 0)} ms | {up(s['median_latency_simulated_ms'], 0)} ms |")
    e = R / "ollama_eviction_check.json"
    if e.exists():
        print("\nEviction check:", json.dumps(json.loads(e.read_text())["summary"]))


if __name__ == "__main__":
    main()
