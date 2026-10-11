import os
import time
import threading
import statistics
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

import psutil

from .models import ModelType, IntentType, QueryResponse, PerformanceMetrics, InferenceMetrics
from .model_registry import ModelRegistry, ModelConfig
from .prompt_manager import PromptManager
from .backends import InferenceBackend, MockBackend, GenerationResult, normalize_tag


@dataclass
class ModelInstance:
    """Represents a loaded model instance"""
    model_name: str
    model_type: ModelType
    is_loaded: bool = False
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    usage_count: int = 0
    error_count: int = 0


def _ms(ns: Optional[int]) -> Optional[float]:
    return ns / 1e6 if ns is not None else None


class ModelManager:
    """Manages model loading, caching, and lifecycle on an inference backend"""

    def __init__(self, max_cache_size: int = 5, backend: Optional[InferenceBackend] = None,
                 registry: Optional[ModelRegistry] = None,
                 default_options: Optional[Dict[str, Any]] = None):
        self.max_cache_size = max_cache_size
        self.backend = backend or MockBackend()
        self.model_registry = registry or ModelRegistry(self.backend)
        self.prompt_manager = PromptManager()
        self.loaded_models: Dict[str, ModelInstance] = {}
        # Reentrant: reload_model and unload_model call helpers that lock too
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.default_options = default_options if default_options is not None else {
            "num_predict": int(os.getenv("ROUTER_NUM_PREDICT", "256"))
        }
        self.last_error: Optional[str] = None
        # Per-thread record of a load that happened during the current request
        self._local = threading.local()

        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []

    def get_model_for_query(self, query: str, intent: IntentType,
                           model_name: Optional[str] = None) -> Optional[ModelConfig]:
        """Get the model to run a query on, loading it if needed"""
        try:
            # If specific model requested, use it
            if model_name:
                if self._load_model(model_name):
                    return self.model_registry.get_model_config(model_name)
                return None

            # Otherwise, get best model for intent
            best_model = self.model_registry.get_best_model_for_intent(intent)
            if not best_model:
                self.logger.warning(f"No suitable model found for intent: {intent}")
                return None

            if self._load_model(best_model.name):
                return self.model_registry.get_model_config(best_model.name)
            return None

        except Exception as e:
            self.logger.error(f"Error getting model for query: {str(e)}")
            self.last_error = str(e)
            return None

    def _load_model(self, model_name: str) -> Optional[ModelInstance]:
        """Load a model on the backend"""
        with self.lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                model_instance = self.loaded_models[model_name]
                model_instance.last_used = time.time()
                model_instance.usage_count += 1
                return model_instance

            config = self.model_registry.get_model_config(model_name)
            info = self.model_registry.get_model_info(model_name)
            if info is not None and not info.is_available and \
                    (info.unavailable_reason or "").startswith("insufficient"):
                # The snapshot may predate a load or unload; re-probe once before refusing
                self._refresh_resources()
                info = self.model_registry.get_model_info(model_name)
            if config is None or info is None:
                self.last_error = f"Unknown model: {model_name}"
                self.logger.error(self.last_error)
                return None
            if not info.is_available:
                self.last_error = f"Model {model_name} is unavailable: {info.unavailable_reason}"
                self.logger.error(self.last_error)
                return None

            # Check if we need to unload a model
            if len(self.loaded_models) >= self.max_cache_size:
                self._unload_least_used_model()

            try:
                self.logger.info(f"Loading model: {model_name} on {self.backend.name} backend")
                load_time = self.backend.load(config)
                self._refresh_resources()

                model_instance = ModelInstance(
                    model_name=model_name,
                    model_type=config.model_type,
                    is_loaded=True,
                    load_time=load_time,
                    last_used=time.time(),
                    usage_count=1
                )
                self.loaded_models[model_name] = model_instance
                self._local.last_load = (model_name, load_time)

                # Update registry
                self.model_registry.update_model_status(model_name, True, load_time)

                self.logger.info(f"Loaded model {model_name} in {load_time:.2f}s")
                return model_instance

            except Exception as e:
                self.last_error = f"Failed to load model {model_name}: {str(e)}"
                self.logger.error(self.last_error)
                return None

    def _unload(self, model_name: str) -> bool:
        """Unload a specific model from the backend and the cache"""
        with self.lock:
            config = self.model_registry.get_model_config(model_name)
            if config is None:
                return False
            self.logger.info(f"Unloading model: {model_name}")
            try:
                ok = self.backend.unload(config)
            except Exception as e:
                self.logger.error(f"Failed to unload {model_name}: {e}")
                ok = False
            self.loaded_models.pop(model_name, None)
            self.model_registry.update_model_status(model_name, False)
            self._refresh_resources()
            return ok

    def _refresh_resources(self):
        """Re-probe VRAM/RAM after the set of resident models changed"""
        try:
            self.model_registry.resource_monitor.snapshot(refresh=True)
        except Exception as e:
            self.logger.debug(f"Resource refresh failed: {e}")

    def _unload_least_used_model(self):
        """Unload the least recently used model"""
        with self.lock:
            if not self.loaded_models:
                return
            lru_model = min(self.loaded_models.values(), key=lambda m: m.last_used or 0)
            self._unload(lru_model.model_name)

    def process_query(self, query: str, intent: IntentType,
                     model_name: Optional[str] = None,
                     context: Optional[List[Dict[str, str]]] = None,
                     options: Optional[Dict[str, Any]] = None,
                     routing_time: Optional[float] = None) -> Optional[QueryResponse]:
        """Process a query using the appropriate model.

        ``routing_time`` (seconds) is the time the caller spent classifying
        and selecting; it is added to the reported total latency.
        """
        start_time = time.perf_counter()
        self.last_error = None
        self._local.last_load = None

        try:
            # Get model (loads it if it is not loaded yet)
            model = self.get_model_for_query(query, intent, model_name)
            if not model:
                self._record_performance_metric(
                    model_name or "unknown", intent, time.perf_counter() - start_time,
                    success=False, error_message=self.last_error, routing_time=routing_time
                )
                return None
            last_load = getattr(self._local, "last_load", None)
            load_s = last_load[1] if last_load and last_load[0] == model.name else None

            # Generate response
            messages = self.prompt_manager.build_messages(intent, model.model_type, query, context)
            gen_options = {**self.default_options, **(options or {})}
            result = self.backend.generate(model, messages, gen_options, intent=intent)
            if (result.load_duration_ns or 0) > 100e6:
                # The backend (re)loaded the model during this call
                self._refresh_resources()

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            metrics = self._build_metrics(result, routing_time, load_s, processing_time)

            with self.lock:
                if model.name in self.loaded_models:
                    self.loaded_models[model.name].last_used = time.time()

            # Record performance
            self._record_performance_metric(
                model.name, intent, processing_time, success=True,
                routing_time=routing_time, result=result
            )

            tokens_used = None
            if result.eval_count is not None:
                tokens_used = (result.prompt_eval_count or 0) + result.eval_count
            elif result.text:
                tokens_used = len(result.text.split())  # Rough estimate for the mock

            return QueryResponse(
                response=result.text,
                model_used=model.name,
                intent_classified=intent,
                routing_decision=None,  # Set by the caller
                processing_time=processing_time,
                tokens_used=tokens_used,
                cost_estimate=processing_time * 0.001,  # Rough estimate
                metrics=metrics
            )

        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.logger.error(f"Error processing query: {self.last_error}")
            self._record_performance_metric(
                model_name or "unknown", intent, time.perf_counter() - start_time,
                success=False, error_message=self.last_error, routing_time=routing_time
            )
            return None

    def _build_metrics(self, result: GenerationResult, routing_time: Optional[float],
                       load_s: Optional[float], processing_time: float) -> InferenceMetrics:
        total = processing_time + (routing_time or 0.0)
        return InferenceMetrics(
            backend=result.backend,
            backend_model=result.backend_model,
            routing_time_ms=routing_time * 1000 if routing_time is not None else None,
            model_load_ms=load_s * 1000 if load_s is not None else None,
            time_to_first_token_ms=result.ttft_s * 1000 if result.ttft_s is not None else None,
            prompt_eval_ms=_ms(result.prompt_eval_duration_ns),
            backend_load_ms=_ms(result.load_duration_ns),
            generation_ms=_ms(result.eval_duration_ns),
            prompt_tokens=result.prompt_eval_count,
            completion_tokens=result.eval_count,
            tokens_per_second=result.tokens_per_second,
            inference_latency_ms=result.wall_time_s * 1000,
            total_latency_ms=total * 1000
        )

    def _record_performance_metric(self, model_name: str, intent: IntentType,
                                 latency: float, success: bool,
                                 error_message: Optional[str] = None,
                                 routing_time: Optional[float] = None,
                                 result: Optional[GenerationResult] = None):
        """Record performance metrics. ``latency`` excludes routing time."""
        try:
            metric = PerformanceMetrics(
                timestamp=time.time(),
                model_name=model_name,
                intent_type=intent,
                latency=latency + (routing_time or 0.0),
                resource_usage={
                    "memory_percent": psutil.virtual_memory().percent,
                    "cpu_percent": psutil.cpu_percent()
                },
                cost=latency * 0.001,  # Rough cost estimate
                success=success,
                error_message=error_message,
                backend=self.backend.name,
                routing_time=routing_time,
                ttft=result.ttft_s if result else None,
                tokens_per_second=result.tokens_per_second if result else None,
                completion_tokens=result.eval_count if result else None
            )

            with self.lock:
                self.performance_metrics.append(metric)
                # Keep only last 1000 metrics
                if len(self.performance_metrics) > 1000:
                    self.performance_metrics = self.performance_metrics[-1000:]

        except Exception as e:
            self.logger.error(f"Error recording performance metric: {str(e)}")

    @staticmethod
    def _summarize(metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        def avg(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else None

        ok = [m for m in metrics if m.success]
        latencies = sorted(m.latency for m in ok)
        return {
            "requests": len(metrics),
            "success_rate": len(ok) / len(metrics),
            "errors": len(metrics) - len(ok),
            "avg_latency": avg(latencies),
            "p50_latency": statistics.median(latencies) if latencies else None,
            "avg_routing_time_ms": avg([m.routing_time * 1000 if m.routing_time is not None else None for m in ok]),
            "avg_ttft_ms": avg([m.ttft * 1000 if m.ttft is not None else None for m in ok]),
            "avg_tokens_per_second": avg([m.tokens_per_second for m in ok]),
            "avg_completion_tokens": avg([m.completion_tokens for m in ok]),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics over the last 100 requests.

        Latencies are in seconds and include routing time when the caller
        supplied it. Averages skip requests where a metric is not reported
        (the mock backend reports no token counts or TTFT).
        """
        with self.lock:
            if not self.performance_metrics:
                return {}
            all_count = len(self.performance_metrics)
            recent_metrics = self.performance_metrics[-100:]  # Last 100 metrics

        summary = self._summarize(recent_metrics)
        stats = {
            "backend": self.backend.name,
            "total_requests": all_count,
            "recent_requests": len(recent_metrics),
            "success_rate": summary["success_rate"],
            "avg_latency": summary["avg_latency"],
            "p50_latency": summary["p50_latency"],
            "avg_routing_time_ms": summary["avg_routing_time_ms"],
            "avg_ttft_ms": summary["avg_ttft_ms"],
            "avg_tokens_per_second": summary["avg_tokens_per_second"],
            "avg_cost": sum(m.cost for m in recent_metrics) / len(recent_metrics),
            "model_performance": {},
            "intent_performance": {}
        }

        by_model: Dict[str, List[PerformanceMetrics]] = {}
        by_intent: Dict[str, List[PerformanceMetrics]] = {}
        for metric in recent_metrics:
            by_model.setdefault(metric.model_name, []).append(metric)
            by_intent.setdefault(metric.intent_type.value, []).append(metric)

        stats["model_performance"] = {name: self._summarize(ms) for name, ms in by_model.items()}
        stats["intent_performance"] = {name: self._summarize(ms) for name, ms in by_intent.items()}
        return stats

    def get_loaded_models(self) -> List[str]:
        """Registry names of models the backend currently holds in memory"""
        try:
            backend_tags = {normalize_tag(t) for t in self.backend.loaded_models()}
        except Exception:
            return list(self.loaded_models.keys())
        if self.backend.name == "mock":
            return list(self.loaded_models.keys())
        return [name for name, cfg in self.model_registry.models.items()
                if cfg.ollama_tag and normalize_tag(cfg.ollama_tag) in backend_tags]

    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        with self.lock:
            if model_name in self.loaded_models or model_name in self.get_loaded_models():
                return self._unload(model_name)
            return False

    def reload_model(self, model_name: str) -> bool:
        """Reload a model"""
        with self.lock:
            # Unload if currently loaded
            if model_name in self.loaded_models:
                self._unload(model_name)

            # Load again
            return self._load_model(model_name) is not None
