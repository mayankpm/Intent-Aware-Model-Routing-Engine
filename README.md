# Logic-Based Routing Engine for Open-Source LLMs

An intelligent routing system that automatically directs user queries to the most suitable open-source Large Language Model (LLM) based on classified intent. The system prioritizes cost-efficiency, latency, and context relevance while maintaining consistency across different models through carefully tuned system prompts.

## 🚀 Features

### Core Capabilities
- **Intent Classification**: Analyzes queries to determine the most appropriate model
- **Intelligent Routing**: Routes queries to the best model based on intent, performance, and resources
- **Custom Reasoning**: Supports specialized domain-specific reasoning engines with auto-selection
- **General Reasoning**: Standard reasoning capabilities for common tasks
- **Model Management**: Dynamic loading, caching, and lifecycle management
- **Performance Optimization**: Cost and latency optimization with resource constraints
- **Consistency Framework**: Maintains uniform tone and personality across all models

### Intent Categories
- **Code Generation**: Programming, debugging, algorithm implementation
- **Creative Writing**: Story generation, content creation, creative tasks
- **General Reasoning**: Standard logical reasoning, problem-solving, analysis
- **Custom Reasoning**: In-house specialized reasoning engines and domain-specific logic
- **Conversation**: Chat, Q&A, general conversation
- **Specialized Tasks**: Translation, summarization, specific domain tasks
- **Multi-Intent**: Complex queries requiring multiple capabilities

### Model Types
- **Small Models** (< 3B parameters): Fast inference, low resource usage
- **Medium Models** (3B-7B parameters): Balanced performance
- **Large Models** (7B+ parameters): High quality, resource intensive
- **Custom Reasoning Engines**: Domain-specific in-house models

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Query Input   │───▶│ Intent Classifier│───▶│ Routing Engine  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Registry │◀───│ Model Manager   │◀───│ Selected Model  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Prompt Manager  │◀───│ Response Output │◀───│ Model Processing│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- 4+ CPU cores (8+ recommended)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/mayankpm/Intent-Aware-Model-Routing-Engine.git
cd Intent-Aware-Model-Routing-Engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

The application will be available at:
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/status

## 🎯 Usage

### Basic Query Processing

```python
import requests

# Process a query
response = requests.post("http://localhost:8000/query", json={
    "query": "Write a Python function to calculate fibonacci numbers",
    "user_id": "user123",
    "auto_custom_reasoning": True
})

print(response.json())
```

### Intent Classification

```python
# Classify intent of a query
response = requests.post("http://localhost:8000/classify-intent", 
                        params={"query": "Explain the concept of recursion"})

print(response.json())
# Output: {"intent": "general_reasoning", "confidence": 0.85, ...}
```

### System Status

```python
# Get system status and statistics
response = requests.get("http://localhost:8000/status")
print(response.json())
```

### Test Endpoint

```python
# Quick test query
response = requests.post("http://localhost:8000/test-query", 
                        params={"query": "Create a story about a robot"})
print(response.json())
```

## 🔧 API Endpoints

### Core Endpoints
- `POST /query` - Process a query through the routing engine
- `POST /test-query` - Quick test query processing
- `POST /classify-intent` - Classify the intent of a query

### Information Endpoints
- `GET /` - Root endpoint with basic info
- `GET /status` - System status and statistics
- `GET /models` - List available models
- `GET /models/{model_name}` - Get specific model information
- `GET /routing-stats` - Routing statistics
- `GET /performance-stats` - Performance statistics

## 🧠 Custom Reasoning

The system supports custom reasoning engines that can be automatically selected for specialized tasks:

### Auto-Selection Logic
- Detects domain-specific keywords and patterns
- Automatically routes to custom reasoning engines when appropriate
- Maintains fallback to general reasoning for standard tasks

### Custom Engine Integration
```python
# Custom reasoning engines are automatically detected and used
response = requests.post("http://localhost:8000/query", json={
    "query": "Apply business logic to optimize the workflow process",
    "auto_custom_reasoning": True
})
```

## 📊 Performance Monitoring

The system provides comprehensive monitoring:

### Metrics Tracked
- Intent classification accuracy
- Model selection patterns
- Response latency
- Resource usage
- Cost estimates
- Success rates

### Accessing Metrics
```python
# Get routing statistics
routing_stats = requests.get("http://localhost:8000/routing-stats").json()

# Get performance statistics
perf_stats = requests.get("http://localhost:8000/performance-stats").json()
```

## 🔧 Configuration

### System Configuration
The system can be configured through the `SystemConfig` class:

```python
from src.models import SystemConfig

config = SystemConfig(
    max_concurrent_requests=10,
    model_cache_size=5,
    enable_auto_custom_reasoning=True,
    custom_reasoning_threshold=0.7,
    cost_optimization_enabled=True
)
```

### Model Registry
Models can be added to the registry in `src/model_registry.py`:

```python
self.models["new-model"] = ModelConfig(
    name="new-model",
    model_type=ModelType.OPEN_SOURCE,
    size=ModelSize.MEDIUM,
    capabilities=[IntentType.CODE_GENERATION, IntentType.GENERAL_REASONING],
    resource_requirements={"min_ram_gb": 8, "cpu_cores": 4},
    performance_metrics={"latency_ms": 1200, "accuracy": 0.85}
)
```

## 🧪 Testing

### Running Tests
```bash
pytest tests/
```

### Example Test Queries
```python
# Code generation
"Write a function to sort a list in Python"

# Creative writing
"Create a short story about a time traveler"

# General reasoning
"Explain the concept of machine learning"

# Custom reasoning
"Apply domain expertise to optimize the business process"

# Conversation
"Hello, how are you today?"

# Specialized tasks
"Translate this text to Spanish"
```

## 🚀 Deployment

### Development
```bash
python run.py
```

### Production
```bash
# Using uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Docker (if Dockerfile is provided)
docker build -t routing-engine .
docker run -p 8000:8000 routing-engine
```

## 📈 Performance Targets

- **Latency**: < 2 seconds for simple queries, < 5 seconds for complex queries
- **Throughput**: 10+ concurrent requests
- **Accuracy**: 90%+ intent classification accuracy
- **Cost**: Optimize for lowest cost per request while maintaining quality

## 🔒 Security Considerations

- Input validation and sanitization
- Rate limiting (can be added)
- Authentication and authorization (can be added)
- Secure model loading and execution
- Privacy protection for user data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Open-source LLM community
- FastAPI for the excellent web framework
- Hugging Face for model management tools
- The AI/ML community for inspiration and best practices

---

**Note**: This is a proof-of-concept implementation. For production use, additional features like authentication, rate limiting, and more robust error handling should be added.
