from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import ModelInfo, ModelType, ModelSize, IntentType
from .backends import InferenceBackend
from .resources import ResourceMonitor, model_fits

# VRAM footprints below were measured with Ollama 0.40.2 on an RTX 3070 Laptop
# GPU: size_vram from /api/ps with each model loaded alone at its default
# context length (tinyllama 2048, phi 2048, mistral 4096). On a machine
# without a GPU the same number is used as the RAM requirement.
SIZE_RANK = {ModelSize.SMALL: 0, ModelSize.MEDIUM: 1, ModelSize.LARGE: 2, ModelSize.CUSTOM: 3}


@dataclass
class ModelConfig:
    """Configuration for a model"""
    name: str
    model_type: ModelType
    size: ModelSize
    capabilities: List[IntentType]
    resource_requirements: Dict[str, any]
    performance_metrics: Dict[str, float]
    model_path: Optional[str] = None
    is_loaded: bool = False
    load_time: Optional[float] = None
    ollama_tag: Optional[str] = None      # backend model that implements this entry
    is_llm: bool = True                   # False for the custom reasoning placeholders
    disabled_reason: Optional[str] = None # set to keep an entry listed but never routable
    think: Optional[bool] = None          # hybrid reasoning models (Qwen3): False = answer directly


class ModelRegistry:
    """Registry for managing available models and their capabilities"""
    
    def __init__(self, backend: Optional[InferenceBackend] = None,
                 resource_monitor: Optional[ResourceMonitor] = None):
        self.backend = backend
        self.resource_monitor = resource_monitor or ResourceMonitor(backend)
        self.models: Dict[str, ModelConfig] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the model registry with available models"""
        
        # Small Models (< 3B parameters)
        self.models["phi-2"] = ModelConfig(
            name="phi-2",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.SMALL,
            capabilities=[
                IntentType.CODE_GENERATION,
                IntentType.GENERAL_REASONING,
                IntentType.CONVERSATION
            ],
            resource_requirements={
                "min_ram_gb": 2.09,
                "min_vram_gb": 2.09,
                "cpu_cores": 2
            },
            performance_metrics={
                "latency_ms": 500,
                "throughput_rps": 10,
                "accuracy": 0.75
            },
            ollama_tag="phi"
        )
        
        self.models["tinyllama"] = ModelConfig(
            name="tinyllama",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.SMALL,
            capabilities=[
                IntentType.CREATIVE_WRITING,
                IntentType.CONVERSATION,
                IntentType.GENERAL_REASONING
            ],
            resource_requirements={
                "min_ram_gb": 0.65,
                "min_vram_gb": 0.65,
                "cpu_cores": 2
            },
            performance_metrics={
                "latency_ms": 600,
                "throughput_rps": 8,
                "accuracy": 0.70
            },
            ollama_tag="tinyllama"
        )
        
        # Medium Models (3B-7B parameters)
        self.models["llama-2-7b"] = ModelConfig(
            name="llama-2-7b",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.MEDIUM,
            capabilities=[
                IntentType.CODE_GENERATION,
                IntentType.CREATIVE_WRITING,
                IntentType.GENERAL_REASONING,
                IntentType.CONVERSATION,
                IntentType.SPECIALIZED_TASKS
            ],
            resource_requirements={
                "min_ram_gb": 8,
                "min_vram_gb": 0,
                "cpu_cores": 4
            },
            performance_metrics={
                "latency_ms": 1500,
                "throughput_rps": 5,
                "accuracy": 0.85
            },
            ollama_tag=None,
            disabled_reason="not installed: duplicates Mistral-7B and is not pulled in Ollama"
        )
        
        self.models["mistral-7b"] = ModelConfig(
            name="mistral-7b",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.MEDIUM,
            capabilities=[
                IntentType.CODE_GENERATION,
                IntentType.CREATIVE_WRITING,
                IntentType.GENERAL_REASONING,
                IntentType.CONVERSATION,
                IntentType.SPECIALIZED_TASKS
            ],
            resource_requirements={
                "min_ram_gb": 4.61,
                "min_vram_gb": 4.61,
                "cpu_cores": 4
            },
            performance_metrics={
                "latency_ms": 1200,
                "throughput_rps": 6,
                "accuracy": 0.88
            },
            ollama_tag="mistral"
        )
        
        # Qwen3 models (2025). They carry no hand-set capabilities or accuracy
        # priors: only the learned routing policies (src/routing_policies.py),
        # which use measured outcomes, can select them. Thinking mode is off.
        # VRAM measured with /api/ps at the default 4096 context.
        self.models["qwen3-1.7b"] = ModelConfig(
            name="qwen3-1.7b",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.SMALL,
            capabilities=[],
            resource_requirements={"min_ram_gb": 1.59, "min_vram_gb": 1.59, "cpu_cores": 2},
            performance_metrics={},
            ollama_tag="qwen3:1.7b",
            think=False
        )

        self.models["qwen3-4b"] = ModelConfig(
            name="qwen3-4b",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.MEDIUM,
            capabilities=[],
            resource_requirements={"min_ram_gb": 2.96, "min_vram_gb": 2.96, "cpu_cores": 4},
            performance_metrics={},
            ollama_tag="qwen3:4b",
            think=False
        )

        # Large Models (7B+ parameters)
        self.models["llama-2-13b"] = ModelConfig(
            name="llama-2-13b",
            model_type=ModelType.OPEN_SOURCE,
            size=ModelSize.LARGE,
            capabilities=[
                IntentType.CODE_GENERATION,
                IntentType.CREATIVE_WRITING,
                IntentType.GENERAL_REASONING,
                IntentType.CONVERSATION,
                IntentType.SPECIALIZED_TASKS,
                IntentType.MULTI_INTENT
            ],
            resource_requirements={
                "min_ram_gb": 16,
                "min_vram_gb": 0,
                "cpu_cores": 8
            },
            performance_metrics={
                "latency_ms": 3000,
                "throughput_rps": 3,
                "accuracy": 0.92
            },
            ollama_tag=None,
            disabled_reason="not installed: a 13B model does not fit in 8 GB of VRAM and is not pulled"
        )
        
        # Custom Reasoning Engines. These are placeholders, not LLMs: no backend
        # implements them except the mock, which returns canned text. With the
        # Ollama backend they are unavailable, so the router never picks them.
        self.models["business-logic-engine"] = ModelConfig(
            name="business-logic-engine",
            model_type=ModelType.CUSTOM_REASONING,
            size=ModelSize.CUSTOM,
            capabilities=[
                IntentType.CUSTOM_REASONING,
                IntentType.GENERAL_REASONING
            ],
            resource_requirements={
                "min_ram_gb": 4,
                "min_vram_gb": 0,
                "cpu_cores": 2
            },
            performance_metrics={
                "latency_ms": 800,
                "throughput_rps": 12,
                "accuracy": 0.95
            },
            is_llm=False
        )
        
        self.models["domain-expert-engine"] = ModelConfig(
            name="domain-expert-engine",
            model_type=ModelType.CUSTOM_REASONING,
            size=ModelSize.CUSTOM,
            capabilities=[
                IntentType.CUSTOM_REASONING,
                IntentType.SPECIALIZED_TASKS
            ],
            resource_requirements={
                "min_ram_gb": 6,
                "min_vram_gb": 0,
                "cpu_cores": 3
            },
            performance_metrics={
                "latency_ms": 1000,
                "throughput_rps": 10,
                "accuracy": 0.90
            },
            is_llm=False
        )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        resources = self.resource_monitor.snapshot()
        return [self._to_model_info(config, resources) for config in self.models.values()]

    def _to_model_info(self, config: ModelConfig, resources: Dict) -> ModelInfo:
        reason = self._unavailable_reason(config, resources)
        return ModelInfo(
            name=config.name,
            model_type=config.model_type,
            size=config.size,
            capabilities=config.capabilities,
            resource_requirements=config.resource_requirements,
            performance_metrics=config.performance_metrics,
            is_available=reason is None,
            load_time=config.load_time,
            is_llm=config.is_llm,
            backend_model=config.ollama_tag,
            unavailable_reason=reason
        )
    
    def get_models_for_intent(self, intent: IntentType) -> List[ModelInfo]:
        """Get models that can handle a specific intent"""
        available_models = self.get_available_models()
        suitable_models = []
        
        for model in available_models:
            if intent in model.capabilities and model.is_available:
                suitable_models.append(model)
        
        return suitable_models
    
    def get_best_model_for_intent(self, intent: IntentType, 
                                 prefer_small: bool = True) -> Optional[ModelInfo]:
        """Get the best model for a specific intent"""
        suitable_models = self.get_models_for_intent(intent)
        
        if not suitable_models:
            return None
        
        # Sort by performance and resource efficiency
        if prefer_small:
            # Prefer smaller models for efficiency
            suitable_models.sort(key=lambda m: (
                SIZE_RANK[m.size],  # Small -> Medium -> Large -> Custom
                -m.performance_metrics.get("accuracy", 0),  # Higher accuracy
                m.performance_metrics.get("latency_ms", float('inf'))  # Lower latency
            ))
        else:
            # Prefer larger models for quality
            suitable_models.sort(key=lambda m: (
                -m.performance_metrics.get("accuracy", 0),  # Higher accuracy first
                m.performance_metrics.get("latency_ms", float('inf'))  # Then lower latency
            ))
        
        return suitable_models[0]
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        if model_name not in self.models:
            return None
        
        return self._to_model_info(self.models[model_name], self.resource_monitor.snapshot())

    def get_fallback_model(self, intent: Optional[IntentType] = None) -> Optional[ModelInfo]:
        """An available model to use when nothing suits ``intent``.

        Tries models for the intent, then general reasoning models, then any
        available model, preferring small ones. Returns None only when no
        model at all is available.
        """
        for candidates in (
            self.get_models_for_intent(intent) if intent else [],
            self.get_models_for_intent(IntentType.GENERAL_REASONING),
            [m for m in self.get_available_models() if m.is_available],
        ):
            if candidates:
                return min(candidates, key=lambda m: (
                    SIZE_RANK[m.size], -m.performance_metrics.get("accuracy", 0)))
        return None

    def _check_model_availability(self, config: ModelConfig) -> bool:
        """Check if a model can run on the active backend with current resources"""
        return self._unavailable_reason(config, self.resource_monitor.snapshot()) is None

    def _unavailable_reason(self, config: ModelConfig, resources: Dict) -> Optional[str]:
        if config.disabled_reason:
            return config.disabled_reason
        if self.backend is not None:
            reason = self.backend.unavailable_reason(config)
            if reason:
                return reason
        try:
            if not model_fits(config, resources):
                return "insufficient free resources (VRAM/RAM/CPU)"
        except Exception:
            # If we can't check resources, assume available
            pass
        return None
    
    def update_model_status(self, model_name: str, is_loaded: bool, load_time: Optional[float] = None):
        """Update the loading status of a model"""
        if model_name in self.models:
            self.models[model_name].is_loaded = is_loaded
            self.models[model_name].load_time = load_time
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get the raw model configuration"""
        return self.models.get(model_name) 