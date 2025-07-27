import time
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .models import (
    QueryRequest, QueryResponse, RoutingDecision, SystemConfig,
    IntentType, ModelInfo
)
from .routing_engine import RoutingEngine
from .model_manager import ModelManager
from .intent_classifier import IntentClassifier
from .model_registry import ModelRegistry
from .prompt_manager import PromptManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoutingApplication:
    """Main application class that orchestrates all components"""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        
        # Initialize components
        self.routing_engine = RoutingEngine(self.config)
        self.model_manager = ModelManager(max_cache_size=self.config.model_cache_size)
        self.intent_classifier = IntentClassifier()
        self.model_registry = ModelRegistry()
        self.prompt_manager = PromptManager()
        
        logger.info("Routing application initialized successfully")
    
    def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query through the routing engine"""
        start_time = time.time()
        
        try:
            # Step 1: Get routing decision
            routing_decision = self.routing_engine.route_query(
                request.query, 
                user_preferences={
                    "preferred_model_size": "small",
                    "prefer_quality": False
                }
            )
            
            # Step 2: Process query with selected model
            query_response = self.model_manager.process_query(
                request.query,
                routing_decision.intent,
                routing_decision.selected_model
            )
            
            if not query_response:
                raise HTTPException(status_code=500, detail="Failed to process query")
            
            # Step 3: Update response with routing decision
            query_response.routing_decision = routing_decision
            
            # Step 4: Calculate total processing time
            total_time = time.time() - start_time
            query_response.processing_time = total_time
            
            logger.info(f"Query processed successfully in {total_time:.2f}s")
            return query_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        try:
            # Get routing statistics
            routing_stats = self.routing_engine.get_routing_stats()
            
            # Get model performance statistics
            model_stats = self.model_manager.get_performance_stats()
            
            # Get available models
            available_models = self.model_registry.get_available_models()
            
            # Get loaded models
            loaded_models = self.model_manager.get_loaded_models()
            
            return {
                "status": "healthy",
                "routing_stats": routing_stats,
                "model_stats": model_stats,
                "available_models": [model.name for model in available_models],
                "loaded_models": loaded_models,
                "config": {
                    "max_concurrent_requests": self.config.max_concurrent_requests,
                    "model_cache_size": self.config.model_cache_size,
                    "enable_auto_custom_reasoning": self.config.enable_auto_custom_reasoning,
                    "cost_optimization_enabled": self.config.cost_optimization_enabled
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }


# Create FastAPI app
app = FastAPI(
    title="Logic-Based Routing Engine for Open-Source LLMs",
    description="An intelligent routing system that directs queries to the most suitable open-source LLM based on classified intent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create application instance
routing_app = RoutingApplication()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Logic-Based Routing Engine for Open-Source LLMs",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the routing engine"""
    return routing_app.process_query(request)


@app.get("/status")
async def get_status():
    """Get system status and statistics"""
    return routing_app.get_system_status()


@app.get("/models")
async def get_models():
    """Get available models"""
    models = routing_app.model_registry.get_available_models()
    return {
        "models": [
            {
                "name": model.name,
                "type": model.model_type.value,
                "size": model.size.value,
                "capabilities": [cap.value for cap in model.capabilities],
                "is_available": model.is_available,
                "performance_metrics": model.performance_metrics
            }
            for model in models
        ]
    }


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    model_info = routing_app.model_registry.get_model_info(model_name)
    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "name": model_info.name,
        "type": model_info.model_type.value,
        "size": model_info.size.value,
        "capabilities": [cap.value for cap in model_info.capabilities],
        "is_available": model_info.is_available,
        "resource_requirements": model_info.resource_requirements,
        "performance_metrics": model_info.performance_metrics
    }


@app.post("/classify-intent")
async def classify_intent(query: str):
    """Classify the intent of a query"""
    try:
        classification = routing_app.intent_classifier.classify_intent(query)
        return {
            "intent": classification.intent.value,
            "confidence": classification.confidence,
            "sub_intents": [intent.value for intent in classification.sub_intents] if classification.sub_intents else None,
            "reasoning": classification.reasoning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/routing-stats")
async def get_routing_stats():
    """Get routing statistics"""
    return routing_app.routing_engine.get_routing_stats()


@app.get("/performance-stats")
async def get_performance_stats():
    """Get performance statistics"""
    return routing_app.model_manager.get_performance_stats()


@app.post("/test-query")
async def test_query(query: str, intent: Optional[str] = None):
    """Test endpoint for quick query processing"""
    try:
        # Create request
        request = QueryRequest(query=query)
        
        # Process query
        response = routing_app.process_query(request)
        
        return {
            "query": query,
            "response": response.response,
            "model_used": response.model_used,
            "intent_classified": response.intent_classified.value,
            "processing_time": response.processing_time,
            "routing_decision": {
                "selected_model": response.routing_decision.selected_model,
                "intent": response.routing_decision.intent.value,
                "confidence": response.routing_decision.confidence,
                "reasoning": response.routing_decision.reasoning,
                "estimated_latency": response.routing_decision.estimated_latency,
                "estimated_cost": response.routing_decision.estimated_cost
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 