"""Run every evaluation item on every model and record outcomes for router research.

For each item it stores the query embedding (all-minilm), the intent the
production classifier assigns, and the legacy router's choice. For each
(item, model) pair it stores whether the answer is correct (research/checks.py),
measured latency, TTFT, Ollama timings, and per-token log-probabilities with
arrival times (used to simulate the probe cascade exactly).

Prompts go through the production path: the PromptManager system prompt for
the classified intent, then Ollama /api/chat. The run is resumable: finished
(item, model) pairs are skipped.

Usage: python -m research.collect --run main [--temperature 0] [--seed 0]
"""
import argparse
import json
import time
from pathlib import Path

from research.checks import score
from src.backends import OllamaBackend
from src.intent_classifier import IntentClassifier
from src.model_registry import ModelRegistry
from src.models import IntentType, ModelType
from src.progress import Progress
from src.prompt_manager import PromptManager
from src.routing_engine import RoutingEngine
from src.backends import MockBackend

ROOT = Path(__file__).parent
MODELS = ["tinyllama", "qwen3-1.7b", "phi-2", "qwen3-4b", "mistral-7b"]   # small to large
LEGACY_MODELS = ["tinyllama", "phi-2", "mistral-7b"]
TOP_K = 5            # top log-probs kept per token
KEEP_TOP_TOKENS = 64  # top-k lists kept for the first N tokens (the probe only needs the start)


def load_items():
    items = []
    for name in ("gsm8k", "mbpp", "mmlu", "custom"):
        with open(ROOT / "data" / f"{name}.jsonl", encoding="utf-8") as f:
            items += [json.loads(l) for l in f]
    return items


def embed(backend, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        r = backend.session.post(f"{backend.base_url}/api/embed",
                                 json={"model": "all-minilm", "input": texts[i:i + batch]}, timeout=120)
        r.raise_for_status()
        out += r.json()["embeddings"]
    return out


class _Fixed:
    """Resources that never bind, so the legacy router reproduces its Ollama routing table."""
    backend = None
    background = False

    def snapshot(self, refresh=False):
        return {"available_ram_gb": 64, "total_ram_gb": 64, "cpu_usage_percent": 0, "available_cpu_cores": 16,
                "disk_usage_percent": 0, "gpu": {"name": "any", "total_vram_gb": 8, "free_vram_gb": 8,
                                                 "used_vram_gb": 0, "source": "fixed"},
                "backend_loaded_vram_gb": {}}


def build_meta(items, backend, meta_path):
    if meta_path.exists():
        return {m["id"]: m for m in map(json.loads, open(meta_path, encoding="utf-8"))}
    clf = IntentClassifier()
    legacy = RoutingEngine(model_registry=ModelRegistry(MockBackend(runnable=LEGACY_MODELS), _Fixed()))
    vectors = embed(backend, [it["prompt"] for it in items])
    meta = {}
    with open(meta_path, "w", encoding="utf-8") as f:
        for it, vec in zip(items, vectors):
            c = clf.classify_intent(it["prompt"])
            m = {"id": it["id"], "source": it["source"], "task": it["task"],
                 "intent": c.intent.value, "intent_confidence": c.confidence,
                 "legacy_choice": legacy.route_query(it["prompt"]).selected_model,
                 "prompt_words": len(it["prompt"].split()), "embedding": vec}
            meta[it["id"]] = m
            f.write(json.dumps(m) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N items of each source (smoke test)")
    args = ap.parse_args()

    out_dir = ROOT / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for line in open(rec_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["item_id"], r["model"]))

    backend = OllamaBackend(keep_alive="30m")
    if not backend.is_reachable():
        raise SystemExit("Ollama is not reachable")
    registry = ModelRegistry(backend)
    prompts = PromptManager()
    items = load_items()
    meta = build_meta(items, backend, ROOT / "data" / "item_meta.jsonl")
    if args.limit:
        per = {}
        items = [it for it in items if per.setdefault(it["source"], []).append(it) or len(per[it["source"]]) <= args.limit]
    (out_dir / "config.json").write_text(json.dumps({
        "temperature": args.temperature, "seed": args.seed, "models": args.models,
        "ollama_version": backend.version, "items": len(items), "top_k": TOP_K,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))

    todo = [(m, it) for m in args.models for it in items if (it["id"], m) not in done]
    print(f"run '{args.run}': {len(items)} items x {len(args.models)} models, {len(done)} done, {len(todo)} to go", flush=True)
    progress = Progress(len(todo) + len(args.models))
    stats = {}
    with open(rec_path, "a", encoding="utf-8") as out:
        for model_name in args.models:
            model = registry.get_model_config(model_name)
            mine = [it for (m, it) in todo if m == model_name]
            backend.unload_all()
            time.sleep(1)
            backend.generate(model, [{"role": "user", "content": "Say hello."}], {"num_predict": 8})
            progress.step(f"{model_name}: warm-up")
            ok = n = 0
            for it in mine:
                intent = meta[it["id"]]["intent"]
                messages = prompts.build_messages(IntentType(intent), ModelType.OPEN_SOURCE, it["prompt"])
                options = {"num_predict": it.get("max_tokens", 256), "temperature": args.temperature, "seed": args.seed}
                try:
                    res = backend.generate(model, messages, options, logprobs=TOP_K)
                    passed = bool(score(res.text, it["check"]))
                    err = None
                except Exception as e:  # record and continue; a failed call counts as wrong
                    res, passed, err = None, False, f"{type(e).__name__}: {e}"
                rec = {"item_id": it["id"], "model": model_name, "passed": passed, "error": err,
                       "temperature": args.temperature, "seed": args.seed}
                if res is not None:
                    rec.update({
                        "response": res.text, "wall_s": res.wall_time_s, "ttft_s": res.ttft_s,
                        "prompt_tokens": res.prompt_eval_count, "prompt_eval_ms": (res.prompt_eval_duration_ns or 0) / 1e6,
                        "eval_count": res.eval_count, "eval_ms": (res.eval_duration_ns or 0) / 1e6,
                        "load_ms": (res.load_duration_ns or 0) / 1e6, "done_reason": res.done_reason,
                        "token_logprobs": [round(x, 5) for x in (res.token_logprobs or [])],
                        "token_top_logprobs": [[round(x, 4) for x in t] for t in (res.token_top_logprobs or [])[:KEEP_TOP_TOKENS]],
                        "token_times_s": [round(x, 4) for x in (res.token_times_s or [])[:KEEP_TOP_TOKENS]],
                    })
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n += 1
                ok += passed
                progress.step(f"{model_name} {it['id']:<10} {'PASS' if passed else 'fail'} | "
                              f"{model_name} acc so far {ok}/{n} = {ok / n:.0%} | "
                              f"{(res.wall_time_s if res else 0):.1f}s")
            stats[model_name] = (ok, n)
    print("\nfinished:", {m: f"{ok}/{n}" for m, (ok, n) in stats.items()}, f"in {Progress.fmt(progress.elapsed())}")


if __name__ == "__main__":
    main()
