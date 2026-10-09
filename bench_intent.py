"""Latency benchmark for intent classification.

Times IntentClassifier.classify_intent (keyword rules + TF-IDF + Multinomial
Naive Bayes) on 2,000 generated queries and prints mean, p50, p90 and p99.

Usage: python bench_intent.py
"""
import platform
import random
import statistics
import time

from src.intent_classifier import IntentClassifier

TEMPLATES = [
    "Write a Python function to {task}",
    "Fix this bug in my {lang} code: {task}",
    "Explain how {topic} works",
    "What is the difference between {topic} and {other}?",
    "Summarize this article about {topic}",
    "Translate this paragraph about {topic} into Spanish",
    "Solve this math problem: {math}",
    "Prove that {math}",
    "Write a short story about {topic}",
    "Give me a poem about {topic}",
    "Analyze the pros and cons of {topic} versus {other}",
    "Step by step, reason about whether {topic} implies {other}",
    "hi",
    "Can you help me plan a trip to {place}?",
]
FILL = {
    "task": ["reverse a linked list", "parse a CSV file", "merge two sorted arrays",
             "debounce an event handler", "read a JSON config", "retry a failed request"],
    "lang": ["Python", "JavaScript", "C++", "Java", "Go"],
    "topic": ["TCP congestion control", "photosynthesis", "inflation", "gradient descent",
              "the French Revolution", "black holes", "vaccines", "blockchain"],
    "other": ["UDP", "respiration", "deflation", "Newton's method", "the American Revolution",
              "neutron stars", "antibiotics", "databases"],
    "math": ["the sum of the first n odd numbers is n squared", "2x + 3 = 11",
             "the integral of x squared from 0 to 3", "sqrt(2) is irrational"],
    "place": ["Tokyo", "Lisbon", "Banff", "Cape Town"],
}


def make_query(rng):
    t = rng.choice(TEMPLATES)
    return t.format(**{k: rng.choice(v) for k, v in FILL.items()})


def main(n=2000, warmup=200, seed=7):
    rng = random.Random(seed)
    clf = IntentClassifier()
    queries = [make_query(rng) for _ in range(n)]
    for q in queries[:warmup]:
        clf.classify_intent(q)
    times_ms = []
    for q in queries:
        t0 = time.perf_counter_ns()
        clf.classify_intent(q)
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)
    times_ms.sort()
    pct = lambda p: times_ms[min(len(times_ms) - 1, int(p * len(times_ms)))]
    print(f"queries: {n} (after {warmup} warm-up calls), seed {seed}")
    print(f"machine: {platform.system()} {platform.release()}, Python {platform.python_version()}")
    print(f"mean {statistics.mean(times_ms):.2f} ms | p50 {pct(0.50):.2f} ms | "
          f"p90 {pct(0.90):.2f} ms | p99 {pct(0.99):.2f} ms | max {times_ms[-1]:.2f} ms")


if __name__ == "__main__":
    main()
