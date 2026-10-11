import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .models import IntentType, RoutingDecision, SystemConfig
from .intent_classifier import IntentClassifier
from .model_registry import ModelRegistry
from .resources import model_fits


@dataclass
class RoutingContext:
    """Context for routing decisions"""
    query: str
    intent: IntentType
    confidence: float
    available_models: list
    system_resources: Dict[str, Any]
    user_preferences: Dict[str, Any]


class RoutingEngine:
    """Intelligent routing engine for selecting the best model for each query"""
    
    def __init__(self, config: Optional[SystemConfig] = None,
                 model_registry: Optional[ModelRegistry] = None,
                 learned_router=None):
        self.config = config or SystemConfig()
        self.intent_classifier = IntentClassifier()
        self.model_registry = model_registry or ModelRegistry()
        self.learned_router = learned_router   # src.learned_router.LearnedRouter, or None for legacy
        self.routing_history = []

    def route_query(self, query: str, user_preferences: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """Route a query to the most suitable model"""
        return self.route_with_plan(query, user_preferences)[0]

    def route_with_plan(self, query: str, user_preferences: Optional[Dict[str, Any]] = None):
        """Route a query. Returns (RoutingDecision, cascade plan or None)."""
        start_time = time.time()

        # Step 1: Classify intent
        intent_classification = self.intent_classifier.classify_intent(query)

        # Learned policy (trained on measured outcomes): used when a trained router is loaded
        if self.learned_router is not None:
            learned = self._route_learned(query, intent_classification)
            if learned is not None:
                self._log_routing_decision(learned[0], time.time() - start_time)
                return learned
        
        # Step 2: Get available models for this intent
        available_models = self.model_registry.get_models_for_intent(intent_classification.intent)
        
        # Step 3: Check custom reasoning auto-selection
        if (self.config.enable_auto_custom_reasoning and 
            self.intent_classifier.should_use_custom_reasoning(query, intent_classification.confidence)):
            # Prefer custom reasoning engines
            custom_models = [m for m in available_models if m.model_type.value == "custom_reasoning"]
            if custom_models:
                available_models = custom_models
        
        # Step 4: Get system resources
        system_resources = self._get_system_resources()
        
        # Step 5: Create routing context
        context = RoutingContext(
            query=query,
            intent=intent_classification.intent,
            confidence=intent_classification.confidence,
            available_models=available_models,
            system_resources=system_resources,
            user_preferences=user_preferences or {}
        )
        
        # Step 6: Make routing decision
        selected_model = self._select_best_model(context)

        # Step 6b: Nothing available for this intent (or nothing fits the
        # resources): fall back to an available model rather than a name that
        # does not exist
        used_fallback = False
        if selected_model is None:
            selected_model = self.model_registry.get_fallback_model(intent_classification.intent)
            used_fallback = selected_model is not None

        # Step 7: Calculate estimates
        estimated_latency = self._estimate_latency(selected_model, context)
        estimated_cost = self._estimate_cost(selected_model, context)
        
        # Step 8: Create routing decision
        routing_decision = RoutingDecision(
            selected_model=selected_model.name if selected_model else "fallback",
            intent=intent_classification.intent,
            confidence=intent_classification.confidence,
            reasoning=self._generate_routing_reasoning(context, selected_model, used_fallback),
            estimated_latency=estimated_latency,
            estimated_cost=estimated_cost,
            fallback_model=self._get_fallback_model(context)
        )
        
        # Step 9: Log routing decision
        self._log_routing_decision(routing_decision, time.time() - start_time)

        return routing_decision, None

    def _route_learned(self, query, classification):
        """Pick by learned P(correct) minus a latency penalty among available models."""
        available = [m.name for m in self.model_registry.get_available_models() if m.is_available and m.is_llm]
        try:
            choice = self.learned_router.choose(query, classification.intent.value, available)
        except Exception as e:  # e.g. embedding model missing: fall back to the legacy policy
            import logging
            logging.getLogger(__name__).warning(f"Learned routing failed ({e}); using legacy policy")
            return None
        if choice is None:
            return None
        lr = self.learned_router
        lat_ms = lr.art["latency_means_s"][lr.models.index(choice.model)] * 1000
        p = choice.p_correct[choice.model]
        reasoning = (f"Learned router ({lr.name}, lambda={lr.lam:g}): P(correct)={p:.2f} for {choice.model}, "
                     f"query difficulty {choice.difficulty:+.2f}; picks the best predicted accuracy minus "
                     f"a latency penalty among available models")
        plan_info = None
        if choice.cascade is not None:
            c = choice.cascade
            plan_info = {"small": c.small, "large": c.large, "k": c.k, "tau": c.tau}
            reasoning += f". Probe cascade: escalate to {c.large} if P(correct) after {c.k} tokens < {c.tau:.2f}"
        others = sorted((m for m in choice.p_correct if m != choice.model and m in available),
                        key=lambda m: -choice.p_correct[m])
        decision = RoutingDecision(
            selected_model=choice.model, intent=classification.intent, confidence=classification.confidence,
            reasoning=reasoning, estimated_latency=lat_ms, estimated_cost=0.0,
            fallback_model=others[0] if others else None, policy=lr.name,
            predicted_success=choice.p_correct, difficulty=choice.difficulty, cascade=plan_info)
        return decision, choice.cascade
    
    def _select_best_model(self, context: RoutingContext) -> Optional[Any]:
        """Select the best model based on routing context"""
        if not context.available_models:
            return None
        
        # Apply user preferences
        preferred_size = context.user_preferences.get("preferred_model_size", "small")
        prefer_quality = context.user_preferences.get("prefer_quality", False)
        
        # Filter models based on preferences
        filtered_models = context.available_models
        
        # Apply cost optimization if enabled
        if self.config.cost_optimization_enabled:
            filtered_models = self._apply_cost_optimization(filtered_models, context)
        
        # Apply resource constraints
        filtered_models = self._apply_resource_constraints(filtered_models, context)
        
        if not filtered_models:
            return None
        
        # Score models based on multiple criteria
        scored_models = []
        for model in filtered_models:
            score = self._calculate_model_score(model, context, prefer_quality)
            scored_models.append((model, score))
        
        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[0][0] if scored_models else None
    
    def _calculate_model_score(self, model: Any, context: RoutingContext, prefer_quality: bool) -> float:
        """Calculate a score for a model based on multiple criteria"""
        score = 0.0
        
        # Base score from performance metrics
        accuracy = model.performance_metrics.get("accuracy", 0.5)
        latency = model.performance_metrics.get("latency_ms", 1000)
        throughput = model.performance_metrics.get("throughput_rps", 1)
        
        # Intent confidence bonus
        intent_bonus = context.confidence * 0.3
        
        # Size preference (smaller models get bonus for efficiency)
        size_bonus = 0.0
        if model.size.value == "small":
            size_bonus = 0.2
        elif model.size.value == "medium":
            size_bonus = 0.1
        
        # Quality vs efficiency trade-off
        if prefer_quality:
            score = accuracy * 0.6 + (1.0 / latency) * 0.2 + intent_bonus + size_bonus
        else:
            score = (1.0 / latency) * 0.4 + throughput * 0.2 + accuracy * 0.3 + intent_bonus + size_bonus
        
        # Custom reasoning bonus for specialized tasks
        if (context.intent == IntentType.CUSTOM_REASONING and 
            model.model_type.value == "custom_reasoning"):
            score += 0.3
        
        return score
    
    def _apply_cost_optimization(self, models: list, context: RoutingContext) -> list:
        """Apply cost optimization to model selection"""
        # In a real implementation, this would consider actual compute costs
        # For now, we'll prefer smaller models for cost efficiency
        return sorted(models, key=lambda m: (
            m.size.value,  # Prefer smaller models
            m.performance_metrics.get("latency_ms", float('inf'))
        ))
    
    def _apply_resource_constraints(self, models: list, context: RoutingContext) -> list:
        """Apply resource constraints to model selection.

        Uses measured free VRAM (plus VRAM held by models the backend can
        evict) when a GPU is present, otherwise free RAM. See resources.model_fits.
        """
        return [model for model in models if model_fits(model, context.system_resources)]
    
    def _estimate_latency(self, model: Any, context: RoutingContext) -> float:
        """Estimate latency for a model"""
        if not model:
            return 5000.0  # Default fallback latency
        
        base_latency = model.performance_metrics.get("latency_ms", 1000)
        
        # Adjust based on query complexity
        query_length = len(context.query)
        complexity_factor = 1.0 + (query_length / 1000) * 0.5
        
        # Adjust based on system load
        system_load = context.system_resources.get("cpu_usage_percent", 50)
        load_factor = 1.0 + (system_load / 100) * 0.3
        
        return base_latency * complexity_factor * load_factor
    
    def _estimate_cost(self, model: Any, context: RoutingContext) -> float:
        """Estimate cost for a model"""
        if not model:
            return 0.01  # Default fallback cost
        
        # Simple cost estimation based on model size and complexity
        base_cost = 0.001  # Base cost per request
        
        if model.size.value == "small":
            size_multiplier = 1.0
        elif model.size.value == "medium":
            size_multiplier = 2.0
        elif model.size.value == "large":
            size_multiplier = 4.0
        else:  # custom
            size_multiplier = 1.5
        
        query_complexity = len(context.query) / 100
        complexity_multiplier = 1.0 + query_complexity * 0.5
        
        return base_cost * size_multiplier * complexity_multiplier
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource information (RAM, CPU, VRAM, loaded models)"""
        try:
            return self.model_registry.resource_monitor.snapshot()
        except Exception:
            # Fallback if probing fails
            return {
                "available_ram_gb": 8.0,
                "total_ram_gb": 16.0,
                "cpu_usage_percent": 50.0,
                "available_cpu_cores": 4,
                "disk_usage_percent": 50.0
            }
    
    def _generate_routing_reasoning(self, context: RoutingContext, selected_model: Any,
                                    used_fallback: bool = False) -> str:
        """Generate human-readable reasoning for routing decision"""
        if not selected_model:
            return "No model is available on the active backend"

        reasoning_parts = []

        if used_fallback:
            reasoning_parts.append(
                f"No available model handles {context.intent.value} within current resources; "
                f"fell back to {selected_model.name}")

        # Intent-based reasoning
        reasoning_parts.append(f"Query classified as {context.intent.value} with {context.confidence:.2f} confidence")
        
        # Model selection reasoning
        if selected_model.model_type.value == "custom_reasoning":
            reasoning_parts.append("Selected custom reasoning engine for specialized domain knowledge")
        elif context.confidence > 0.8:
            reasoning_parts.append(f"High confidence ({context.confidence:.2f}) in intent classification")
        else:
            reasoning_parts.append(f"Moderate confidence ({context.confidence:.2f}) in intent classification")
        
        # Performance reasoning
        accuracy = selected_model.performance_metrics.get("accuracy", 0)
        latency = selected_model.performance_metrics.get("latency_ms", 1000)
        reasoning_parts.append(f"Model accuracy: {accuracy:.2f}, estimated latency: {latency}ms")
        
        return ". ".join(reasoning_parts)
    
    def _get_fallback_model(self, context: RoutingContext) -> Optional[str]:
        """Get fallback model name"""
        # Simple fallback logic - prefer smaller models
        fallback_models = [m for m in context.available_models if m.size.value == "small"]
        if fallback_models:
            return fallback_models[0].name
        fallback = self.model_registry.get_fallback_model(context.intent)
        return fallback.name if fallback else None
    
    def _log_routing_decision(self, decision: RoutingDecision, processing_time: float):
        """Log routing decision for monitoring"""
        log_entry = {
            "timestamp": time.time(),
            "model": decision.selected_model,
            "intent": decision.intent.value,
            "confidence": decision.confidence,
            "latency_estimate": decision.estimated_latency,
            "cost_estimate": decision.estimated_cost,
            "processing_time": processing_time
        }
        self.routing_history.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {}
        
        recent_decisions = self.routing_history[-100:]  # Last 100 decisions
        
        stats = {
            "total_decisions": len(self.routing_history),
            "recent_decisions": len(recent_decisions),
            "avg_confidence": sum(d["confidence"] for d in recent_decisions) / len(recent_decisions),
            "avg_latency_estimate": sum(d["latency_estimate"] for d in recent_decisions) / len(recent_decisions),
            "avg_cost_estimate": sum(d["cost_estimate"] for d in recent_decisions) / len(recent_decisions),
            "model_usage": {}
        }
        
        # Calculate model usage statistics
        for decision in recent_decisions:
            model = decision["model"]
            stats["model_usage"][model] = stats["model_usage"].get(model, 0) + 1
        
        return stats 