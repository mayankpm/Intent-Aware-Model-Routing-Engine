import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import logging

from .models import ModelInfo, ModelType, IntentType, QueryResponse, PerformanceMetrics
from .model_registry import ModelRegistry
from .prompt_manager import PromptManager


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


class ModelManager:
    """Manages model loading, caching, and lifecycle"""
    
    def __init__(self, max_cache_size: int = 5):
        self.max_cache_size = max_cache_size
        self.model_registry = ModelRegistry()
        self.prompt_manager = PromptManager()
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.model_cache: Dict[str, Any] = {}  # Actual model objects
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
    
    def get_model_for_query(self, query: str, intent: IntentType, 
                           model_name: Optional[str] = None) -> Optional[Any]:
        """Get a model instance for processing a query"""
        start_time = time.time()
        
        try:
            # If specific model requested, use it
            if model_name:
                model_instance = self._load_model(model_name)
                if model_instance:
                    return self._get_model_from_cache(model_name)
            
            # Otherwise, get best model for intent
            best_model = self.model_registry.get_best_model_for_intent(intent)
            if not best_model:
                self.logger.warning(f"No suitable model found for intent: {intent}")
                return None
            
            model_instance = self._load_model(best_model.name)
            if model_instance:
                return self._get_model_from_cache(best_model.name)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting model for query: {str(e)}")
            self._record_performance_metric(
                model_name or "unknown", intent, time.time() - start_time, 
                success=False, error_message=str(e)
            )
            return None
    
    def _load_model(self, model_name: str) -> Optional[ModelInstance]:
        """Load a model into memory"""
        with self.lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                model_instance = self.loaded_models[model_name]
                model_instance.last_used = time.time()
                model_instance.usage_count += 1
                return model_instance
            
            # Check if we need to unload a model
            if len(self.loaded_models) >= self.max_cache_size:
                self._unload_least_used_model()
            
            # Load the model
            try:
                self.logger.info(f"Loading model: {model_name}")
                load_start = time.time()
                
                # In a real implementation, this would load the actual model
                # For this POC, we'll simulate model loading
                model_instance = ModelInstance(
                    model_name=model_name,
                    model_type=self._get_model_type(model_name),
                    is_loaded=True,
                    load_time=time.time() - load_start,
                    last_used=time.time(),
                    usage_count=1
                )
                
                self.loaded_models[model_name] = model_instance
                self.model_cache[model_name] = self._create_mock_model(model_name)
                
                # Update registry
                self.model_registry.update_model_status(
                    model_name, True, model_instance.load_time
                )
                
                self.logger.info(f"Successfully loaded model: {model_name}")
                return model_instance
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {str(e)}")
                return None
    
    def _unload_least_used_model(self):
        """Unload the least recently used model"""
        if not self.loaded_models:
            return
        
        # Find least recently used model
        lru_model = min(
            self.loaded_models.values(),
            key=lambda m: m.last_used or 0
        )
        
        model_name = lru_model.model_name
        self.logger.info(f"Unloading least used model: {model_name}")
        
        # Remove from cache
        if model_name in self.model_cache:
            del self.model_cache[model_name]
        
        # Remove from loaded models
        del self.loaded_models[model_name]
        
        # Update registry
        self.model_registry.update_model_status(model_name, False)
    
    def _get_model_from_cache(self, model_name: str) -> Optional[Any]:
        """Get a model from the cache"""
        return self.model_cache.get(model_name)
    
    def _get_model_type(self, model_name: str) -> ModelType:
        """Get the model type for a model name"""
        model_info = self.model_registry.get_model_info(model_name)
        if model_info:
            return model_info.model_type
        return ModelType.OPEN_SOURCE  # Default
    
    def _create_mock_model(self, model_name: str) -> Any:
        """Create a mock model for demonstration purposes"""
        # In a real implementation, this would load actual models
        # For this POC, we'll create a mock model that simulates responses
        
        class MockModel:
            def __init__(self, name: str):
                self.name = name
                self.prompt_manager = PromptManager()
            
            def generate_response(self, query: str, intent: IntentType, 
                               model_type: ModelType, context: Optional[List[Dict[str, str]]] = None) -> str:
                """Generate a mock response based on intent and model type"""
                
                # Generate appropriate prompt
                prompt = self.prompt_manager.generate_prompt(intent, model_type, query, context)
                
                # Generate mock response based on intent
                if intent == IntentType.CODE_GENERATION:
                    return f"[{self.name}] Here's a code solution for your query: '{query}'\n\n```python\ndef example_function():\n    # Your code here\n    pass\n```"
                
                elif intent == IntentType.CREATIVE_WRITING:
                    return f"[{self.name}] Here's a creative response to: '{query}'\n\nOnce upon a time, in a world of endless possibilities..."
                
                elif intent == IntentType.GENERAL_REASONING:
                    return f"[{self.name}] Let me analyze this step by step: '{query}'\n\n1. First, let's understand the problem\n2. Then, we'll apply logical reasoning\n3. Finally, we'll reach a conclusion"
                
                elif intent == IntentType.CUSTOM_REASONING:
                    return f"[{self.name}] Applying specialized domain knowledge to: '{query}'\n\nBased on domain expertise and specialized methodologies..."
                
                elif intent == IntentType.CONVERSATION:
                    return f"[{self.name}] Hello! I'd be happy to chat about: '{query}'\n\nThat's an interesting topic. Let me share some thoughts..."
                
                elif intent == IntentType.SPECIALIZED_TASKS:
                    return f"[{self.name}] Processing specialized task: '{query}'\n\nApplying specialized techniques and methodologies..."
                
                else:
                    return f"[{self.name}] Here's my response to: '{query}'\n\nI understand your request and will provide a helpful response."
        
        return MockModel(model_name)
    
    def process_query(self, query: str, intent: IntentType, 
                     model_name: Optional[str] = None) -> Optional[QueryResponse]:
        """Process a query using the appropriate model"""
        start_time = time.time()
        
        try:
            # Get model
            model = self.get_model_for_query(query, intent, model_name)
            if not model:
                return None
            
            # Get model info
            model_info = self.model_registry.get_model_info(model.name)
            if not model_info:
                return None
            
            # Generate response
            response = model.generate_response(query, intent, model_info.model_type)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Record performance
            self._record_performance_metric(
                model.name, intent, processing_time, success=True
            )
            
            # Create response object
            return QueryResponse(
                response=response,
                model_used=model.name,
                intent_classified=intent,
                routing_decision=None,  # Will be set by routing engine
                processing_time=processing_time,
                tokens_used=len(response.split()),  # Rough estimate
                cost_estimate=processing_time * 0.001  # Rough estimate
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self._record_performance_metric(
                model_name or "unknown", intent, time.time() - start_time,
                success=False, error_message=str(e)
            )
            return None
    
    def _record_performance_metric(self, model_name: str, intent: IntentType,
                                 latency: float, success: bool, 
                                 error_message: Optional[str] = None):
        """Record performance metrics"""
        import psutil
        
        try:
            # Get resource usage
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            
            metric = PerformanceMetrics(
                timestamp=time.time(),
                model_name=model_name,
                intent_type=intent,
                latency=latency,
                resource_usage={
                    "memory_percent": memory_usage,
                    "cpu_percent": cpu_usage
                },
                cost=latency * 0.001,  # Rough cost estimate
                success=success,
                error_message=error_message
            )
            
            self.performance_metrics.append(metric)
            
            # Keep only last 1000 metrics
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error recording performance metric: {str(e)}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_metrics:
            return {}
        
        recent_metrics = self.performance_metrics[-100:]  # Last 100 metrics
        
        stats = {
            "total_requests": len(self.performance_metrics),
            "recent_requests": len(recent_metrics),
            "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            "avg_latency": sum(m.latency for m in recent_metrics) / len(recent_metrics),
            "avg_cost": sum(m.cost for m in recent_metrics) / len(recent_metrics),
            "model_performance": {},
            "intent_performance": {}
        }
        
        # Calculate model performance
        for metric in recent_metrics:
            model = metric.model_name
            if model not in stats["model_performance"]:
                stats["model_performance"][model] = {
                    "requests": 0,
                    "success_rate": 0,
                    "avg_latency": 0,
                    "errors": 0
                }
            
            stats["model_performance"][model]["requests"] += 1
            if metric.success:
                stats["model_performance"][model]["success_rate"] += 1
            else:
                stats["model_performance"][model]["errors"] += 1
        
        # Calculate averages
        for model_stats in stats["model_performance"].values():
            total = model_stats["requests"]
            model_stats["success_rate"] /= total
            model_stats["avg_latency"] = sum(
                m.latency for m in recent_metrics 
                if m.model_name == model_stats.get("model_name", "")
            ) / total
        
        return stats
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        with self.lock:
            if model_name in self.loaded_models:
                self._unload_least_used_model()
                return True
            return False
    
    def reload_model(self, model_name: str) -> bool:
        """Reload a model"""
        with self.lock:
            # Unload if currently loaded
            if model_name in self.loaded_models:
                self._unload_least_used_model()
            
            # Load again
            return self._load_model(model_name) is not None 