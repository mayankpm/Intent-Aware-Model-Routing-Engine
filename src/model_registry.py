from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import psutil

from .models import ModelInfo, ModelType, ModelSize, IntentType


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


class ModelRegistry:
    """Registry for managing available models and their capabilities"""
    
    def __init__(self):
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
                "min_ram_gb": 4,
                "min_vram_gb": 0,
                "cpu_cores": 2
            },
            performance_metrics={
                "latency_ms": 500,
                "throughput_rps": 10,
                "accuracy": 0.75
            }
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
                "min_ram_gb": 3,
                "min_vram_gb": 0,
                "cpu_cores": 2
            },
            performance_metrics={
                "latency_ms": 600,
                "throughput_rps": 8,
                "accuracy": 0.70
            }
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
            }
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
                "min_ram_gb": 8,
                "min_vram_gb": 0,
                "cpu_cores": 4
            },
            performance_metrics={
                "latency_ms": 1200,
                "throughput_rps": 6,
                "accuracy": 0.88
            }
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
            }
        )
        
        # Custom Reasoning Engines
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
            }
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
            }
        )
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        available_models = []
        
        for model_name, config in self.models.items():
            # Check if system meets requirements
            is_available = self._check_model_availability(config)
            
            model_info = ModelInfo(
                name=config.name,
                model_type=config.model_type,
                size=config.size,
                capabilities=config.capabilities,
                resource_requirements=config.resource_requirements,
                performance_metrics=config.performance_metrics,
                is_available=is_available,
                load_time=config.load_time
            )
            available_models.append(model_info)
        
        return available_models
    
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
                m.size.value,  # Small -> Medium -> Large -> Custom
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
        
        config = self.models[model_name]
        is_available = self._check_model_availability(config)
        
        return ModelInfo(
            name=config.name,
            model_type=config.model_type,
            size=config.size,
            capabilities=config.capabilities,
            resource_requirements=config.resource_requirements,
            performance_metrics=config.performance_metrics,
            is_available=is_available,
            load_time=config.load_time
        )
    
    def _check_model_availability(self, config: ModelConfig) -> bool:
        """Check if a model can be loaded given current system resources"""
        try:
            # Get current system resources
            available_ram = psutil.virtual_memory().available / (1024**3)  # GB
            cpu_count = psutil.cpu_count()
            
            # Check if system meets minimum requirements
            required_ram = config.resource_requirements.get("min_ram_gb", 0)
            required_cpu = config.resource_requirements.get("cpu_cores", 1)
            
            return available_ram >= required_ram and cpu_count >= required_cpu
        except Exception:
            # If we can't check resources, assume available
            return True
    
    def update_model_status(self, model_name: str, is_loaded: bool, load_time: Optional[float] = None):
        """Update the loading status of a model"""
        if model_name in self.models:
            self.models[model_name].is_loaded = is_loaded
            self.models[model_name].load_time = load_time
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get the raw model configuration"""
        return self.models.get(model_name) 