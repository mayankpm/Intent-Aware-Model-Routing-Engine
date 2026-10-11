"""Download fixed random samples of GSM8K, MBPP and MMLU into research/data/.

Sources (licenses): GSM8K (MIT) from github.com/openai/grade-school-math;
MBPP sanitized test split (CC BY 4.0) and MMLU test split (MIT) via the
Hugging Face datasets-server API. Sampling uses a fixed seed, so re-running
reproduces the same items.

Usage: python -m research.fetch_benchmarks [--n 60] [--seed 2026]
"""
import argparse
import json
import random
import time
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def get(url, **params):
    for attempt in range(8):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            time.sleep(0.5)  # stay under the datasets-server rate limit
            return r
        wait = float(r.headers.get("Retry-After", 0) or 0) or min(60, 3 * 2 ** attempt)
        print(f"  HTTP {r.status_code}, retrying in {wait:.0f}s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


def hf_rows(dataset, config, split, total):
    rows = []
    for offset in range(0, total, 100):
        r = get(ROWS_URL, dataset=dataset, config=config, split=split, offset=offset, length=100)
        rows += [x["row"] for x in r.json()["rows"]]
    return rows


def gsm8k(n, rng):
    lines = get(GSM8K_URL).text.strip().splitlines()
    items = [json.loads(l) for l in lines]
    out = []
    for i in sorted(rng.sample(range(len(items)), n)):
        q, a = items[i]["question"], items[i]["answer"]
        answer = float(a.split("####")[-1].strip().replace(",", ""))
        out.append({
            "id": f"gsm8k-{i}", "source": "gsm8k", "task": "math",
            "prompt": q + "\nSolve it step by step, then give the final answer on the last line as 'Answer: <number>'.",
            "max_tokens": 384,
            "check": {"type": "number", "answer": answer},
        })
    return out


def mbpp(n, rng):
    rows = hf_rows("google-research-datasets/mbpp", "sanitized", "test", 257)
    out = []
    for i in sorted(rng.sample(range(len(rows)), n)):
        r = rows[i]
        tests = r["test_list"]
        entry = None
        for t in tests:
            m = __import__("re").search(r"assert\s+(?:not\s+)?\(?\s*(?:math\.isclose\()?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", t)
            if m:
                entry = m.group(1)
                break
        out.append({
            "id": f"mbpp-{r['task_id']}", "source": "mbpp", "task": "code",
            "prompt": (f"{r['prompt']}\nYour code should pass these tests:\n" + "\n".join(tests) +
                       "\nReturn only the Python code in a ```python code block."),
            "max_tokens": 384,
            "check": {"type": "python", "setup": "\n".join(r.get("test_imports") or []), "tests": tests, "entry": entry},
            "reference": r["code"],
        })
    return out


def mmlu(n, rng):
    total = 14042
    out = []
    for i in sorted(rng.sample(range(total), n)):
        r = get(ROWS_URL, dataset="cais/mmlu", config="all", split="test", offset=i, length=1).json()["rows"][0]["row"]
        opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        out.append({
            "id": f"mmlu-{i}", "source": "mmlu", "task": "knowledge", "subject": r["subject"],
            "prompt": (f"{r['question']}\n{opts}\nThink briefly, then give the letter of the correct option "
                       "on the last line as 'Answer: <letter>'."),
            "max_tokens": 256,
            "check": {"type": "choice", "answer": "ABCD"[r["answer"]]},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for name, fn in (("gsm8k", gsm8k), ("mbpp", mbpp), ("mmlu", mmlu)):
        if (DATA / f"{name}.jsonl").exists() and not args.force:
            print(f"{name}: already present (use --force to refetch)")
            continue
        items = fn(args.n, random.Random(f"{args.seed}-{name}"))
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"{name}: {len(items)} items")


if __name__ == "__main__":
    main()
