"""Measure whether Ollama keeps the router's models loaded together.

The learned router calls the embedding model on every query, then one of
several LLMs. If Ollama holds fewer models than that at once, models evict
each other and every switch pays a reload. This alternates embed -> generate
across the routed LLMs and records Ollama's own load_duration and which
models are resident after each call.

Usage: python -m research.eviction_check
Writes results/ollama_eviction_check.json.
"""
import json
from pathlib import Path

import requests

BASE = "http://127.0.0.1:11434"
LLMS = ["qwen3:1.7b", "phi", "qwen3:4b-instruct-2507-q4_K_M"]
OUT = Path(__file__).resolve().parent.parent / "results" / "ollama_eviction_check.json"


def main(rounds=2):
    version = requests.get(f"{BASE}/api/version", timeout=5).json()["version"]
    rows = []
    for rnd in range(rounds):
        for m in LLMS:
            e = requests.post(f"{BASE}/api/embed", json={"model": "all-minilm", "input": ["hello"]}, timeout=120).json()
            g = requests.post(f"{BASE}/api/chat", json={"model": m, "messages": [{"role": "user", "content": "Say hi"}],
                                                        "stream": False, "options": {"num_predict": 4}}, timeout=300).json()
            loaded = [x["name"] for x in requests.get(f"{BASE}/api/ps", timeout=5).json()["models"]]
            rows.append({"round": rnd, "model": m, "embed_load_ms": e.get("load_duration", 0) / 1e6,
                         "model_load_ms": g.get("load_duration", 0) / 1e6, "loaded_after": loaded})
            print(f"round {rnd} {m:<32} load {rows[-1]['model_load_ms']:8.0f} ms | resident: {loaded}", flush=True)
    summary = {"ollama_version": version, "max_models_resident": max(len(r["loaded_after"]) for r in rows),
               "reloads_over_1s": sum(r["model_load_ms"] > 1000 for r in rows),
               "reload_ms_range": [min(r["model_load_ms"] for r in rows if r["model_load_ms"] > 1000),
                                   max(r["model_load_ms"] for r in rows if r["model_load_ms"] > 1000)]
               if any(r["model_load_ms"] > 1000 for r in rows) else None}
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
