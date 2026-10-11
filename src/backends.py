"""Inference backends.

An InferenceBackend runs a model that the registry knows about. Two
implementations exist:

- OllamaBackend: real local inference through the Ollama HTTP API.
- MockBackend: canned responses, so tests and CI run without a GPU.

Select one with the ROUTER_BACKEND env var (``ollama`` or ``mock``). When
Ollama is requested but not reachable, ``create_backend`` falls back to the
mock and records why, so the API can report it in ``/status``.
"""
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import requests

from .models import IntentType

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def normalize_tag(tag: str) -> str:
    """Ollama reports untagged models as ``name:latest``."""
    return tag if ":" in tag else f"{tag}:latest"


@dataclass
class GenerationResult:
    """Text plus timing for one generation.

    Durations ending in ``_ns`` come from Ollama itself. ``ttft_s`` and
    ``wall_time_s`` are measured by the client around the HTTP call.
    """
    text: str
    model_name: str                       # registry name, e.g. "mistral-7b"
    backend: str                          # "ollama" or "mock"
    backend_model: Optional[str] = None   # model that actually ran, as reported by the backend
    wall_time_s: float = 0.0
    ttft_s: Optional[float] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration_ns: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration_ns: Optional[int] = None
    load_duration_ns: Optional[int] = None
    total_duration_ns: Optional[int] = None
    done_reason: Optional[str] = None
    # Filled when log-probabilities are requested: one entry per generated token
    token_logprobs: Optional[List[float]] = None
    token_top_logprobs: Optional[List[List[float]]] = None
    token_times_s: Optional[List[float]] = None  # client-side arrival time of each token
    aborted: bool = False                        # generation stopped early by `stop_when`

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.eval_count and self.eval_duration_ns:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tokens_per_second"] = self.tokens_per_second
        return d


class InferenceBackend(ABC):
    """Runs models. ``model`` arguments are registry ModelConfig objects."""

    name: str = "base"

    @abstractmethod
    def can_run(self, model) -> bool:
        """Whether this backend can run the given registry model right now."""

    def unavailable_reason(self, model) -> Optional[str]:
        return None if self.can_run(model) else f"backend '{self.name}' cannot run {model.name}"

    @abstractmethod
    def generate(self, model, messages: List[Dict[str, str]],
                 options: Optional[Dict[str, Any]] = None,
                 intent: Optional[IntentType] = None, logprobs: int = 0,
                 stop_when: Optional[Callable[["GenerationResult"], bool]] = None) -> GenerationResult:
        """Generate a reply to chat ``messages`` (system/user/assistant dicts).

        ``logprobs`` > 0 records each token's log-probability and the top
        ``logprobs`` alternatives. ``stop_when`` is called after every token
        with the partial result; returning True aborts generation (used by the
        probe cascade).
        """

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Sentence embeddings for the learned router. Backends without one raise."""
        raise NotImplementedError(f"backend '{self.name}' has no embedding model")

    @abstractmethod
    def load(self, model) -> float:
        """Load the model and return the wall time it took, in seconds."""

    @abstractmethod
    def unload(self, model) -> bool:
        """Unload the model. Returns True if the backend accepted the request."""

    @abstractmethod
    def loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Models the backend currently holds in memory, keyed by backend tag."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name}


class MockBackend(InferenceBackend):
    """Returns canned text. Used for tests, CI and when Ollama is down.

    ``run_non_llm`` controls whether the custom reasoning engines (which have
    no real implementation) count as runnable. With ``llm_only`` set, the mock
    accepts the same models Ollama would, which lets tests reproduce the
    Ollama routing table without a GPU.
    """

    name = "mock"

    def __init__(self, run_non_llm: bool = True, runnable: Optional[List[str]] = None,
                 token_logprob: Optional[Dict[str, float]] = None):
        self.run_non_llm = run_non_llm
        self.runnable = set(runnable) if runnable is not None else None
        # Per-model constant token log-probability reported when logprobs are requested (tests)
        self.token_logprob = token_logprob or {}
        self._loaded: Dict[str, float] = {}

    def can_run(self, model) -> bool:
        if self.runnable is not None:
            return model.name in self.runnable
        if not model.is_llm:
            return self.run_non_llm
        return True

    def unavailable_reason(self, model) -> Optional[str]:
        if self.can_run(model):
            return None
        if self.runnable is not None:
            return "not in this mock backend's runnable set"
        return "non-LLM placeholder engine; disabled in this mock configuration"

    def generate(self, model, messages, options=None, intent=None, logprobs=0, stop_when=None) -> GenerationResult:
        start = time.perf_counter()
        query = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        text = self._canned_response(model.name, query, intent)
        result = GenerationResult(text=text, model_name=model.name, backend=self.name,
                                  backend_model=f"mock:{model.name}", done_reason="stop")
        if logprobs or stop_when:
            lp = self.token_logprob.get(model.name, -0.1)
            result.token_logprobs, result.token_top_logprobs, result.token_times_s = [], [], []
            for _ in text.split():
                result.token_logprobs.append(lp)
                result.token_top_logprobs.append([lp])
                result.token_times_s.append(time.perf_counter() - start)
                if stop_when and stop_when(result):
                    result.aborted = True
                    result.done_reason = "aborted"
                    break
        result.wall_time_s = time.perf_counter() - start
        return result

    def embed(self, texts):
        """Deterministic stand-in vectors (hash-seeded), so routing tests run without a model."""
        import hashlib
        import random
        out = []
        for t in texts:
            rng = random.Random(hashlib.sha256(t.encode()).digest())
            out.append([rng.gauss(0, 1) for _ in range(384)])
        return out

    def load(self, model) -> float:
        self._loaded[model.name] = time.time()
        return 0.0

    def unload(self, model) -> bool:
        return self._loaded.pop(model.name, None) is not None

    def loaded_models(self) -> Dict[str, Dict[str, Any]]:
        return {f"mock:{name}": {"loaded_at": t} for name, t in self._loaded.items()}

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "run_non_llm": self.run_non_llm,
                "note": "canned responses, no model inference"}

    @staticmethod
    def _canned_response(name: str, query: str, intent: Optional[IntentType]) -> str:
        if intent == IntentType.CODE_GENERATION:
            return f"[{name}] Here's a code solution for your query: '{query}'\n\n```python\ndef example_function():\n    # Your code here\n    pass\n```"
        elif intent == IntentType.CREATIVE_WRITING:
            return f"[{name}] Here's a creative response to: '{query}'\n\nOnce upon a time, in a world of endless possibilities..."
        elif intent == IntentType.GENERAL_REASONING:
            return f"[{name}] Let me analyze this step by step: '{query}'\n\n1. First, let's understand the problem\n2. Then, we'll apply logical reasoning\n3. Finally, we'll reach a conclusion"
        elif intent == IntentType.CUSTOM_REASONING:
            return f"[{name}] Applying specialized domain knowledge to: '{query}'\n\nBased on domain expertise and specialized methodologies..."
        elif intent == IntentType.CONVERSATION:
            return f"[{name}] Hello! I'd be happy to chat about: '{query}'\n\nThat's an interesting topic. Let me share some thoughts..."
        elif intent == IntentType.SPECIALIZED_TASKS:
            return f"[{name}] Processing specialized task: '{query}'\n\nApplying specialized techniques and methodologies..."
        return f"[{name}] Here's my response to: '{query}'\n\nI understand your request and will provide a helpful response."


class OllamaBackend(InferenceBackend):
    """Real inference through a local Ollama server (``/api/chat``)."""

    name = "ollama"

    def __init__(self, base_url: Optional[str] = None, keep_alive: Optional[str] = None,
                 timeout: float = 300.0, tags_ttl: float = 30.0):
        self.base_url = (base_url or os.getenv("OLLAMA_HOST_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
        self.keep_alive = keep_alive or os.getenv("ROUTER_KEEP_ALIVE", "10m")
        self.timeout = timeout
        self.tags_ttl = tags_ttl
        self.session = requests.Session()
        self._tags: Optional[set] = None
        self._tags_fetched_at = 0.0
        self.version: Optional[str] = None

    # -- server info ---------------------------------------------------------

    def is_reachable(self, timeout: float = 2.0) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/version", timeout=timeout)
            r.raise_for_status()
            self.version = r.json().get("version")
            return True
        except requests.RequestException:
            return False

    def pulled_tags(self, refresh: bool = False) -> set:
        if refresh or self._tags is None or time.time() - self._tags_fetched_at > self.tags_ttl:
            try:
                r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
                r.raise_for_status()
                self._tags = {m["name"] for m in r.json().get("models", [])}
            except requests.RequestException as e:
                logger.warning(f"Could not list Ollama models: {e}")
                self._tags = self._tags or set()
            self._tags_fetched_at = time.time()
        return self._tags

    # -- InferenceBackend ----------------------------------------------------

    def can_run(self, model) -> bool:
        tag = getattr(model, "ollama_tag", None)
        return bool(tag) and normalize_tag(tag) in self.pulled_tags()

    def unavailable_reason(self, model) -> Optional[str]:
        tag = getattr(model, "ollama_tag", None)
        if not tag:
            return "no Ollama model mapped (non-LLM placeholder)" if not model.is_llm else "no Ollama model mapped"
        if normalize_tag(tag) not in self.pulled_tags():
            return f"Ollama model '{tag}' is not pulled"
        return None

    def generate(self, model, messages, options=None, intent=None, logprobs=0, stop_when=None) -> GenerationResult:
        payload = {
            "model": model.ollama_tag,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": options or {},
        }
        if getattr(model, "think", None) is not None:
            payload["think"] = model.think  # Qwen3-style hybrid models: False = answer directly
        track = bool(logprobs or stop_when)
        if track:
            payload["logprobs"] = True
            payload["top_logprobs"] = max(1, logprobs)
        start = time.perf_counter()
        ttft = None
        parts: List[str] = []
        final: Dict[str, Any] = {}
        partial = GenerationResult(text="", model_name=model.name, backend=self.name,
                                   backend_model=model.ollama_tag)
        if track:
            partial.token_logprobs, partial.token_top_logprobs, partial.token_times_s = [], [], []
        with self.session.post(f"{self.base_url}/api/chat", json=payload,
                               stream=True, timeout=self.timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama /api/chat returned {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama error: {chunk['error']}")
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    parts.append(piece)
                if track:
                    now = time.perf_counter() - start
                    for lp in chunk.get("logprobs") or []:
                        partial.token_logprobs.append(lp["logprob"])
                        partial.token_top_logprobs.append([t["logprob"] for t in lp.get("top_logprobs") or []])
                        partial.token_times_s.append(now)
                if chunk.get("done"):
                    final = chunk
                    break
                if stop_when and partial.token_logprobs:
                    partial.text = "".join(parts)
                    if stop_when(partial):
                        # Leaving the `with` block closes the connection; Ollama stops generating
                        partial.aborted = True
                        break
        wall = time.perf_counter() - start
        if partial.aborted:
            partial.text = "".join(parts)
            partial.wall_time_s, partial.ttft_s, partial.done_reason = wall, ttft, "aborted"
            partial.eval_count = len(partial.token_logprobs)
            return partial
        return GenerationResult(
            text="".join(parts),
            token_logprobs=partial.token_logprobs,
            token_top_logprobs=partial.token_top_logprobs,
            token_times_s=partial.token_times_s,
            model_name=model.name,
            backend=self.name,
            backend_model=final.get("model", model.ollama_tag),
            wall_time_s=wall,
            ttft_s=ttft,
            prompt_eval_count=final.get("prompt_eval_count"),
            prompt_eval_duration_ns=final.get("prompt_eval_duration"),
            eval_count=final.get("eval_count"),
            eval_duration_ns=final.get("eval_duration"),
            load_duration_ns=final.get("load_duration"),
            total_duration_ns=final.get("total_duration"),
            done_reason=final.get("done_reason"),
        )

    def embed(self, texts):
        model = os.getenv("ROUTER_EMBED_MODEL", "all-minilm")
        r = self.session.post(f"{self.base_url}/api/embed",
                              json={"model": model, "input": list(texts), "keep_alive": self.keep_alive},
                              timeout=60)
        r.raise_for_status()
        return r.json()["embeddings"]

    def load(self, model) -> float:
        # A generate request with no prompt loads the model and keeps it resident.
        start = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/generate",
                              json={"model": model.ollama_tag, "keep_alive": self.keep_alive},
                              timeout=self.timeout)
        r.raise_for_status()
        return time.perf_counter() - start

    def unload(self, model) -> bool:
        r = self.session.post(f"{self.base_url}/api/generate",
                              json={"model": model.ollama_tag, "keep_alive": 0},
                              timeout=60)
        return r.status_code == 200

    def unload_all(self) -> None:
        for tag in list(self.loaded_models()):
            self.session.post(f"{self.base_url}/api/generate",
                              json={"model": tag, "keep_alive": 0}, timeout=60)

    def loaded_models(self) -> Dict[str, Dict[str, Any]]:
        try:
            r = self.session.get(f"{self.base_url}/api/ps", timeout=5)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Could not query Ollama /api/ps: {e}")
            return {}
        return {
            m["name"]: {
                "size_bytes": m.get("size"),
                "size_vram_bytes": m.get("size_vram"),
                "context_length": m.get("context_length"),
                "expires_at": m.get("expires_at"),
            }
            for m in r.json().get("models", [])
        }

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "url": self.base_url, "version": self.version,
                "keep_alive": self.keep_alive, "pulled_models": sorted(self.pulled_tags())}


@dataclass
class BackendSelection:
    """Which backend was asked for, which one is active, and why."""
    requested: str
    active: str
    reason: str
    backend: InferenceBackend = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {"requested": self.requested, "active": self.active, "reason": self.reason,
                **{k: v for k, v in self.backend.describe().items() if k != "name"}}


def create_backend(requested: Optional[str] = None,
                   ollama_url: Optional[str] = None) -> BackendSelection:
    """Build the backend named by ``requested`` or the ROUTER_BACKEND env var.

    ``ollama`` (the default) uses Ollama if it answers, otherwise the mock.
    ``mock`` always uses the mock.
    """
    requested = (requested or os.getenv("ROUTER_BACKEND") or "ollama").strip().lower()
    if requested not in ("ollama", "mock"):
        raise ValueError(f"ROUTER_BACKEND must be 'ollama' or 'mock', got '{requested}'")

    if requested == "mock":
        logger.info("Inference backend: mock (ROUTER_BACKEND=mock). Responses are canned text.")
        return BackendSelection("mock", "mock", "requested", MockBackend())

    backend = OllamaBackend(base_url=ollama_url)
    if backend.is_reachable():
        logger.info(f"Inference backend: Ollama {backend.version} at {backend.base_url}, "
                    f"models: {sorted(backend.pulled_tags())}")
        return BackendSelection("ollama", "ollama", "Ollama reachable", backend)

    reason = f"Ollama not reachable at {backend.base_url}; falling back to mock (canned responses)"
    logger.warning(f"Inference backend: MOCK. {reason}")
    return BackendSelection("ollama", "mock", reason, MockBackend())
