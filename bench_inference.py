"""Inference benchmark: real local LLMs through Ollama, driven by the router.

For each run and each model:
  1. unload every Ollama model, so the next call is a cold start;
  2. one warm-up call on a fixed prompt: its load time and TTFT are reported
     separately as the cold start;
  3. VRAM: Ollama's size_vram for the model, and the GPU's used memory
     (NVML) minus what was used before the load;
  4. the fixed prompt set (5 per intent, 30 total): each prompt goes through
     RoutingEngine.route_query (classification + selection, timed as router
     overhead) and then ModelManager.process_query on the model under test,
     with fixed num_predict, temperature 0 and a fixed seed.

Then a routed pass sends the same prompts through the full application
(the router picks the model), with all models unloaded first.

Per model it reports tokens/s (eval_count / eval_duration), TTFT (client
measured, plus Ollama's prompt eval time), end-to-end latency p50/p90/p99,
VRAM and router overhead as a share of total latency. Prints tables and
writes everything, including per-request records, to results/.

Usage: python bench_inference.py [--runs 3] [--num-predict 128]
"""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import psutil

from src.backends import create_backend
from src.main import RoutingApplication
from src.models import IntentType, QueryRequest
from src.progress import Progress
from src.resources import GIB, GpuProbe

PROMPTS = {
    IntentType.CODE_GENERATION: [
        "Write a Python function that checks whether a string is a palindrome.",
        "Write a JavaScript function that removes duplicate values from an array.",
        "Write a SQL query that returns the ten most recent orders from an orders table.",
        "Fix this Python code so it does not crash on an empty list: def avg(xs): return sum(xs) / len(xs)",
        "Write a Python class for a stack with push, pop and peek methods.",
    ],
    IntentType.CREATIVE_WRITING: [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Write a poem about the first snow of winter.",
        "Describe a busy night market in a city that floats on the sea.",
        "Write the opening paragraph of a mystery novel set on a train.",
        "Write a dialogue between a cat and a dog who must share a house.",
    ],
    IntentType.GENERAL_REASONING: [
        "Explain why the sky is blue.",
        "Explain how vaccines train the immune system.",
        "Compare renting and buying a home for someone who may move in two years.",
        "Explain why ice floats on water.",
        "Analyze the main causes of inflation.",
    ],
    IntentType.CUSTOM_REASONING: [
        "Apply business logic to decide whether a refund request older than 30 days should be approved.",
        "Design a workflow process for onboarding a new employee in a small company.",
        "Use domain expertise to outline a procedure for handling a failed bank transfer.",
        "Recommend a business process to reduce support ticket backlog.",
        "Apply industry best practices to plan an inventory restocking workflow.",
    ],
    IntentType.CONVERSATION: [
        "Hello, how are you today?",
        "Hi! What should I cook for dinner tonight?",
        "Hey, can you recommend a good book to read this weekend?",
        "Hello there, what's a fun hobby to pick up?",
        "Hi, I'm bored. Want to chat about space?",
    ],
    IntentType.SPECIALIZED_TASKS: [
        "Translate into French: The meeting has been moved to Thursday afternoon.",
        "Summarize in one sentence: Solar panels convert sunlight into electricity. They are increasingly cheap and are being installed on homes and businesses worldwide.",
        "Classify the sentiment of this review as positive or negative: The battery died after two days and support never answered.",
        "Extract the names of people from this sentence: Maria met John and Priya at the station in Lisbon.",
        "Translate into Spanish: Where is the nearest train station?",
    ],
}
PROMPT_COUNT = sum(len(p) for p in PROMPTS.values())
WARMUP_PROMPT = "Say hello in one short sentence."
BENCH_MODELS = ["tinyllama", "phi-2", "mistral-7b"]


def pct(values, p):
    """Nearest-rank percentile, same method as bench_intent.py."""
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def wait_until_unloaded(app, timeout=60):
    backend = app.backend
    backend.unload_all()
    app.model_manager.loaded_models.clear()
    deadline = time.time() + timeout
    while backend.loaded_models() and time.time() < deadline:
        time.sleep(0.5)
    time.sleep(1.0)  # let the driver release memory before reading the baseline


def machine_info(app, gpu):
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "cpu_logical_cores": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / GIB, 2),
        "ram_available_gb_at_start": round(psutil.virtual_memory().available / GIB, 2),
        "python": platform.python_version(),
        "ollama_version": app.backend.version,
        "gpu": None,
    }
    g = gpu.read()
    if g:
        info["gpu"] = {"name": g["name"], "total_vram_gb": round(g["total_vram_gb"], 2)}
        try:
            import pynvml
            driver = pynvml.nvmlSystemGetDriverVersion()
            info["gpu"]["driver"] = driver if isinstance(driver, str) else driver.decode()
        except Exception:
            pass
    tags = app.backend.session.get(f"{app.backend.base_url}/api/tags", timeout=10).json()["models"]
    info["ollama_models"] = {
        m["name"]: {"digest": m["digest"][:12], "size_bytes": m["size"],
                    "parameter_size": m["details"].get("parameter_size"),
                    "quantization": m["details"].get("quantization_level")}
        for m in tags
    }
    return info


def request_record(prompt, intent, routing_s, resp):
    m = resp.metrics
    return {
        "intent_label": intent.value,
        "prompt": prompt,
        "classified_intent": resp.intent_classified.value,
        "model": resp.model_used,
        "backend_model": m.backend_model,
        "routing_ms": routing_s * 1000,
        "ttft_ms": m.time_to_first_token_ms,
        "prompt_eval_ms": m.prompt_eval_ms,
        "prompt_tokens": m.prompt_tokens,
        "backend_load_ms": m.backend_load_ms,
        "model_load_ms": m.model_load_ms,
        "completion_tokens": m.completion_tokens,
        "generation_ms": m.generation_ms,
        "tokens_per_second": m.tokens_per_second,
        "inference_ms": m.inference_latency_ms,
        "total_ms": m.total_latency_ms,
        "response_chars": len(resp.response),
    }


def summarize(records, num_predict):
    total = [r["total_ms"] for r in records]
    routing = [r["routing_ms"] for r in records]
    eval_tokens = sum(r["completion_tokens"] for r in records)
    eval_ms = sum(r["generation_ms"] for r in records)
    return {
        "requests": len(records),
        "tokens_per_second": eval_tokens / (eval_ms / 1000),
        "tokens_per_second_median": statistics.median(r["tokens_per_second"] for r in records),
        "completion_tokens_mean": eval_tokens / len(records),
        "hit_num_predict": sum(1 for r in records if r["completion_tokens"] >= num_predict),
        "ttft_ms_p50": pct([r["ttft_ms"] for r in records], 0.50),
        "ttft_ms_p90": pct([r["ttft_ms"] for r in records], 0.90),
        "prompt_eval_ms_p50": pct([r["prompt_eval_ms"] for r in records], 0.50),
        "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in records),
        "latency_ms_p50": pct(total, 0.50),
        "latency_ms_p90": pct(total, 0.90),
        "latency_ms_p99": pct(total, 0.99),
        "routing_ms_p50": pct(routing, 0.50),
        "routing_ms_p99": pct(routing, 0.99),
        "router_share_pct": 100 * sum(routing) / sum(total),
    }


def bench_model(app, gpu, model_name, num_predict, seed, progress, run_label):
    config = app.model_registry.get_model_config(model_name)
    options = {"num_predict": num_predict, "temperature": 0, "seed": seed}

    wait_until_unloaded(app)
    before = gpu.read()
    warm = app.backend.generate(config, [{"role": "user", "content": WARMUP_PROMPT}], options)
    after = gpu.read()
    progress.step(f"{run_label} {model_name} cold start + warm-up ({warm.wall_time_s:.1f}s)")
    ps = app.backend.loaded_models()
    tag_info = next((v for k, v in ps.items() if k.split(":")[0] == config.ollama_tag), {})
    cold = {
        "load_ms": warm.load_duration_ns / 1e6 if warm.load_duration_ns is not None else None,
        "ttft_ms": warm.ttft_s * 1000 if warm.ttft_s is not None else None,
        "wall_ms": warm.wall_time_s * 1000,
        "vram_ollama_gb": tag_info.get("size_vram_bytes", 0) / GIB if tag_info else None,
        "vram_nvml_delta_gb": (after["used_vram_gb"] - before["used_vram_gb"]) if before and after else None,
        "context_length": tag_info.get("context_length"),
    }
    # Register the already-resident model with the manager so timed requests don't include a load call
    app.model_manager._load_model(model_name)

    records = []
    for intent, prompts in PROMPTS.items():
        for prompt in prompts:
            t0 = time.perf_counter()
            decision = app.routing_engine.route_query(prompt)
            routing_s = time.perf_counter() - t0
            resp = app.model_manager.process_query(prompt, decision.intent, model_name,
                                                   options=options, routing_time=routing_s)
            if resp is None:
                raise RuntimeError(f"{model_name} failed: {app.model_manager.last_error}")
            rec = request_record(prompt, intent, routing_s, resp)
            rec["router_choice"] = decision.selected_model
            records.append(rec)
            progress.step(f"{run_label} {model_name} prompt {len(records)}/{PROMPT_COUNT} "
                          f"({rec['total_ms']:.0f} ms, {rec['tokens_per_second']:.0f} tok/s)")
    return cold, records


def bench_routed(app, num_predict, seed, progress, run_label):
    wait_until_unloaded(app)
    records = []
    for intent, prompts in PROMPTS.items():
        for prompt in prompts:
            resp = app.process_query(QueryRequest(query=prompt, max_tokens=num_predict, temperature=0))
            records.append(request_record(prompt, intent, resp.metrics.routing_time_ms / 1000, resp))
            progress.step(f"{run_label} routed prompt {len(records)}/{PROMPT_COUNT} -> {resp.model_used} "
                          f"({records[-1]['total_ms']:.0f} ms)")
    return records


def print_tables(runs, num_predict):
    print(f"\nPer-model results (num_predict={num_predict}, temperature 0, {len(runs[0]['models'][BENCH_MODELS[0]]['requests'])} prompts)")
    hdr = ("| Run | Model | tok/s | TTFT p50 | Prompt eval p50 | E2E p50 | E2E p90 | E2E p99 | "
           "Router p50 | Router share | Mean tokens | Cold load | Cold TTFT | VRAM (Ollama) | VRAM (NVML delta) |")
    print(hdr)
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        for name in BENCH_MODELS:
            s, c = run["models"][name]["summary"], run["models"][name]["cold_start"]
            print(f"| {run['run']} | {name} | {s['tokens_per_second']:.1f} | {s['ttft_ms_p50']:.0f} ms | "
                  f"{s['prompt_eval_ms_p50']:.0f} ms | {s['latency_ms_p50']:.0f} ms | {s['latency_ms_p90']:.0f} ms | "
                  f"{s['latency_ms_p99']:.0f} ms | {s['routing_ms_p50']:.2f} ms | {s['router_share_pct']:.3f}% | "
                  f"{s['completion_tokens_mean']:.0f} | {c['load_ms']:.0f} ms | {c['ttft_ms']:.0f} ms | "
                  f"{c['vram_ollama_gb']:.2f} GiB | {c['vram_nvml_delta_gb']:.2f} GiB |")
    print("\nRouted pass (router picks the model)")
    print("| Run | Requests per model | E2E p50 | E2E p90 | E2E p99 | Router share | Requests that loaded a model (> 500 ms) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for run in runs:
        r = run["routed"]
        print(f"| {run['run']} | {r['model_counts']} | {r['summary']['latency_ms_p50']:.0f} ms | "
              f"{r['summary']['latency_ms_p90']:.0f} ms | {r['summary']['latency_ms_p99']:.0f} ms | "
              f"{r['summary']['router_share_pct']:.3f}% | {r['reloads']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--num-predict", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    selection = create_backend("ollama")
    if selection.active != "ollama":
        raise SystemExit(f"Ollama backend required: {selection.reason}")
    app = RoutingApplication(backend_selection=selection)
    gpu = GpuProbe()
    for name in BENCH_MODELS:
        info = app.model_registry.get_model_info(name)
        if not info.is_available:
            raise SystemExit(f"{name} unavailable: {info.unavailable_reason}")

    started = datetime.now()
    result = {
        "benchmark": "bench_inference.py",
        "started": started.isoformat(timespec="seconds"),
        "settings": {"runs": args.runs, "num_predict": args.num_predict, "temperature": 0,
                     "seed": args.seed, "prompts_per_model": sum(len(p) for p in PROMPTS.values()),
                     "prompts_per_intent": {i.value: len(p) for i, p in PROMPTS.items()},
                     "keep_alive": app.backend.keep_alive,
                     "percentiles": "nearest rank; with 30 samples p99 is the maximum"},
        "machine": machine_info(app, gpu),
        "runs": [],
    }
    print(json.dumps(result["machine"], indent=1))

    # Per run: each model does 1 warm-up + every prompt, then the routed pass does every prompt
    progress = Progress(args.runs * (len(BENCH_MODELS) * (1 + PROMPT_COUNT) + PROMPT_COUNT))
    for run in range(1, args.runs + 1):
        run_label = f"run {run}/{args.runs}"
        run_result = {"run": run, "models": {}}
        for name in BENCH_MODELS:
            cold, records = bench_model(app, gpu, name, args.num_predict, args.seed, progress, run_label)
            run_result["models"][name] = {"cold_start": cold,
                                          "summary": summarize(records, args.num_predict),
                                          "requests": records}
        routed = bench_routed(app, args.num_predict, args.seed, progress, run_label)
        counts = {}
        for r in routed:
            counts[r["model"]] = counts.get(r["model"], 0) + 1
        run_result["routed"] = {
            "model_counts": counts,
            "reloads": sum(1 for r in routed if (r["backend_load_ms"] or 0) > 500 or (r["model_load_ms"] or 0) > 500),
            "summary": summarize(routed, args.num_predict),
            "requests": routed,
        }
        result["runs"].append(run_result)

    result["finished"] = datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"inference_{started.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(result, indent=1))
    print_tables(result["runs"], args.num_predict)
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
