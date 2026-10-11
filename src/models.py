from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class IntentType(str, Enum):
    """Intent classification categories"""
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    GENERAL_REASONING = "general_reasoning"
    CUSTOM_REASONING = "custom_reasoning"
    CONVERSATION = "conversation"
    SPECIALIZED_TASKS = "specialized_tasks"
    MULTI_INTENT = "multi_intent"


class ModelSize(str, Enum):
    """Model size categories"""
    SMALL = "small"  # < 3B parameters
    MEDIUM = "medium"  # 3B-7B parameters
    LARGE = "large"  # 7B+ parameters
    CUSTOM = "custom"  # Custom reasoning engines


class ModelType(str, Enum):
    """Model types"""
    OPEN_SOURCE = "open_source"
    CUSTOM_REASONING = "custom_reasoning"
    HYBRID = "hybrid"


class QueryRequest(BaseModel):
    """Incoming query request"""
    query: str = Field(..., description="User query text")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[List[Dict[str, str]]] = Field(None, description="Conversation context")
    force_model: Optional[str] = Field(None, description="Force specific model selection")
    auto_custom_reasoning: bool = Field(True, description="Enable auto-selection for custom reasoning")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate (Ollama num_predict)")
    temperature: Optional[float] = Field(None, ge=0.0, description="Sampling temperature")


class IntentClassification(BaseModel):
    """Intent classification result"""
    intent: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    sub_intents: Optional[List[IntentType]] = Field(None, description="Multiple intents for complex queries")
    reasoning: str = Field(..., description="Explanation for classification")


class ModelInfo(BaseModel):
    """Model information and capabilities"""
    name: str
    model_type: ModelType
    size: ModelSize
    capabilities: List[IntentType]
    resource_requirements: Dict[str, Any]
    performance_metrics: Dict[str, float]
    is_available: bool = True
    load_time: Optional[float] = None
    is_llm: bool = True
    backend_model: Optional[str] = Field(None, description="Backend model tag, e.g. the Ollama tag")
    unavailable_reason: Optional[str] = None


class RoutingDecision(BaseModel):
    """Routing decision result"""
    selected_model: str
    intent: IntentType
    confidence: float
    reasoning: str
    estimated_latency: float
    estimated_cost: float
    fallback_model: Optional[str] = None


class InferenceMetrics(BaseModel):
    """Measured timings for one query. Fields are None when the backend does not report them."""
    backend: str = Field(..., description="Backend that ran the query: 'ollama' or 'mock'")
    backend_model: Optional[str] = Field(None, description="Model the backend reports it ran")
    routing_time_ms: Optional[float] = Field(None, description="Intent classification plus model selection")
    model_load_ms: Optional[float] = Field(None, description="Explicit load before generation, if one happened")
    time_to_first_token_ms: Optional[float] = Field(None, description="Client-measured, request sent to first content chunk")
    prompt_eval_ms: Optional[float] = Field(None, description="Backend-reported prompt processing time")
    backend_load_ms: Optional[float] = Field(None, description="Backend-reported load time inside the generate call")
    generation_ms: Optional[float] = Field(None, description="Backend-reported token generation time")
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tokens_per_second: Optional[float] = Field(None, description="completion_tokens / generation time")
    inference_latency_ms: Optional[float] = Field(None, description="Client-measured generate call")
    total_latency_ms: Optional[float] = Field(None, description="Routing plus load plus inference")


class QueryResponse(BaseModel):
    """Response from the routing engine"""
    response: str
    model_used: str
    intent_classified: IntentType
    routing_decision: Optional[RoutingDecision] = None
    processing_time: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    metrics: Optional[InferenceMetrics] = None


class PerformanceMetrics(BaseModel):
    """Performance monitoring metrics"""
    timestamp: datetime
    model_name: str
    intent_type: IntentType
    latency: float
    resource_usage: Dict[str, float]
    cost: float
    success: bool
    error_message: Optional[str] = None
    backend: Optional[str] = None
    routing_time: Optional[float] = None
    ttft: Optional[float] = None
    tokens_per_second: Optional[float] = None
    completion_tokens: Optional[int] = None


class SystemConfig(BaseModel):
    """System configuration"""
    max_concurrent_requests: int = 10
    model_cache_size: int = 5
    default_timeout: float = 30.0
    enable_auto_custom_reasoning: bool = True
    custom_reasoning_threshold: float = 0.7
    cost_optimization_enabled: bool = True 