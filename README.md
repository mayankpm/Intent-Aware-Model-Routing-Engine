# Logic-Based Routing Engine for Open-Source LLMs

An intelligent routing system that automatically directs user queries to the most suitable open-source Large Language Model (LLM) based on classified intent, then runs the query on that model locally through [Ollama](https://ollama.com). The system prioritizes cost-efficiency, latency, and context relevance while maintaining consistency across different models through carefully tuned system prompts.

## 🚀 Features

### Core Capabilities
- **Intent Classification**: Analyzes queries to determine the most appropriate model
- **Intelligent Routing**: Routes queries to the best model based on intent, performance, and resources (free VRAM and RAM are measured, not assumed)
- **Real Local Inference**: Runs the selected model through Ollama (4-bit quantized TinyLlama 1.1B, Phi-2 2.7B and Mistral 7B), with a mock backend for tests and CI
- **Custom Reasoning**: Routing hooks for domain-specific reasoning engines with auto-selection (the two engines in the registry are placeholders; see below)
- **General Reasoning**: Standard reasoning capabilities for common tasks
- **Model Management**: Loading, unloading and LRU caching on the backend (Ollama `keep_alive` and `/api/ps`)
- **Measured Metrics**: Each response reports routing time, time to first token, tokens/s and total latency
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

### Models in the registry

| Registry name | Ollama tag | Status |
|---|---|---|
| `tinyllama` | `tinyllama` (1.1B, Q4_0) | runs on Ollama |
| `phi-2` | `phi` (2.7B, Q4_0) | runs on Ollama |
| `mistral-7b` | `mistral` (7.2B, Q4_K_M) | runs on Ollama |
| `llama-2-7b` | none | listed but disabled: not pulled, duplicates Mistral-7B |
| `llama-2-13b` | none | listed but disabled: a 13B model does not fit in 8 GB of VRAM |
| `business-logic-engine` | none | non-LLM placeholder: only the mock backend runs it (canned text) |
| `domain-expert-engine` | none | non-LLM placeholder: only the mock backend runs it (canned text) |

Disabled and placeholder entries are reported as unavailable (with a reason) on the Ollama backend, so the router never selects them.

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
- Python 3.10+ (tested on 3.12.5)
- [Ollama](https://ollama.com) for real inference, with the models pulled:
  `ollama pull tinyllama`, `ollama pull phi`, `ollama pull mistral` (about 6.6 GB on disk)
- An NVIDIA GPU with 8 GB of VRAM runs all three models fully on the GPU; without one, Ollama runs them on the CPU
- Without Ollama the app still runs, on the mock backend (canned responses)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/mayankpm/Intent-Aware-Model-Routing-Engine.git
cd Intent-Aware-Model-Routing-Engine
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the application with Ollama:
```bash
ROUTER_BACKEND=ollama uvicorn src.main:app --port 8000
# PowerShell: $env:ROUTER_BACKEND="ollama"; uvicorn src.main:app --port 8000
```
or `python run.py`. `/status` shows which backend is active; if Ollama is not reachable the app logs a warning and uses the mock backend.

The application will be available at:
- **Main API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/status

## 🎯 Usage

### Basic Query Processing

```bash
curl -s -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to reverse a linked list", "max_tokens": 128}'
```

The response names the model that ran and its measured metrics:

```json
{
  "response": "...",
  "model_used": "phi-2",
  "intent_classified": "code_generation",
  "metrics": {
    "backend": "ollama", "backend_model": "phi",
    "routing_time_ms": ..., "time_to_first_token_ms": ..., "prompt_eval_ms": ...,
    "completion_tokens": ..., "tokens_per_second": ..., "total_latency_ms": ...
  },
  "routing_decision": {"selected_model": "phi-2", "reasoning": "...", ...}
}
```

Optional request fields: `force_model` (skip selection, e.g. `"mistral-7b"`), `max_tokens`, `temperature`, `context`.

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

The router supports custom reasoning engines that can be automatically selected for specialized tasks. The two engines in the registry (`business-logic-engine`, `domain-expert-engine`) are placeholders with no model behind them: only the mock backend runs them, returning canned text. On the Ollama backend they are unavailable, and `custom_reasoning` queries fall back to an available LLM (the reasoning string says so).

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
- Model selection patterns and intent confidence
- Routing time (classification plus selection)
- Time to first token, prompt eval time, tokens/s (from Ollama's own timing fields)
- End-to-end latency
- Resource usage (RAM, CPU, VRAM)
- Cost estimates (rough, not a real cost model)
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

### Environment Variables

| Variable | Default | Meaning |
|---|---|---|
| `ROUTER_BACKEND` | `ollama` | `ollama` (falls back to mock if unreachable) or `mock` |
| `OLLAMA_HOST_URL` | `http://127.0.0.1:11434` | Ollama server |
| `ROUTER_KEEP_ALIVE` | `10m` | How long Ollama keeps a model loaded after use |
| `ROUTER_NUM_PREDICT` | `256` | Default maximum tokens per response |

### Model Registry
Models can be added to the registry in `src/model_registry.py`:

```python
self.models["new-model"] = ModelConfig(
    name="new-model",
    model_type=ModelType.OPEN_SOURCE,
    size=ModelSize.MEDIUM,
    capabilities=[IntentType.CODE_GENERATION, IntentType.GENERAL_REASONING],
    resource_requirements={"min_vram_gb": 4.6, "min_ram_gb": 4.6, "cpu_cores": 4},
    performance_metrics={"latency_ms": 1200, "accuracy": 0.85},
    ollama_tag="new-model:tag"   # must be pulled in Ollama to be routable
)
```

`min_vram_gb` is checked against free VRAM plus VRAM held by models Ollama can evict; without a GPU, `min_ram_gb` is checked against free RAM.

## 🧪 Testing

### Running Tests
```bash
pytest                 # everything; Ollama tests skip if Ollama or a model is missing
pytest -m "not ollama" # mock backend only, no GPU needed (CI)
pytest -m ollama       # real inference on tinyllama, phi and mistral
```

`python test_demo.py` exercises a running server and prints the backend, model and metrics for each query.

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

These are targets, not measurements. The measured numbers are below.

## ⏱️ Measured: Intent Classification Latency

`bench_intent.py` times `IntentClassifier.classify_intent` (keyword rules plus TF-IDF and Multinomial Naive Bayes) on 2,000 generated queries after 200 warm-up calls.

```bash
python bench_intent.py
```

Three runs on a Windows laptop (Python 3, scikit-learn 1.9):

| Run | Mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| 1 | 0.82 ms | 0.78 ms | 0.90 ms | 1.31 ms |
| 2 | 0.83 ms | 0.79 ms | 0.90 ms | 1.32 ms |
| 3 | 0.83 ms | 0.80 ms | 0.90 ms | 1.27 ms |

This covers intent scoring only. It does not include loading or running any LLM.

## ⚡ Measured: Local Inference

`bench_inference.py` runs the three Ollama models through the real request path: `RoutingEngine.route_query` (classification plus selection, timed as router overhead), then `ModelManager.process_query` against Ollama's `/api/chat`.

```bash
python bench_inference.py --runs 3 --num-predict 128
```

Raw data, including every request: [`results/inference_20261011_104012.json`](results/inference_20261011_104012.json).

### Machine and setup

- **Hardware:** ASUS laptop, AMD Ryzen 9 5900HS, 16 GB RAM, NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM, driver 595.79), Windows 11 (10.0.22631), on AC power
- **Power mode:** ASUS **Silent** profile. Right after the run, `nvidia-smi` reported an enforced GPU power limit of 55 W, with the software thermal-slowdown and power-cap throttle reasons both active (GPU at 81 °C). These numbers describe a throttled GPU; see the note below the tables.
- **Software:** Ollama 0.40.2, Python 3.12.5. About 6 GiB of RAM was free at the start.
- **Models (all 4-bit, all fully on the GPU):**

| Registry name | Ollama tag | Parameters | Quantization | Context | VRAM loaded (Ollama `size_vram`) |
|---|---|---|---|---:|---:|
| tinyllama | `tinyllama:latest` (2644915ede35) | 1.1B | Q4_0 | 2048 | 0.65 GiB |
| phi-2 | `phi:latest` (e2fd6321a5fe) | 2.7B | Q4_0 | 2048 | 2.09 GiB |
| mistral-7b | `mistral:latest` (6577803aa9a0) | 7.2B | Q4_K_M | 4096 | 4.62 GiB |

### Method

- **Prompts:** 30 fixed prompts, 5 per intent (code, creative, general reasoning, custom reasoning, conversation, specialized tasks). Every model runs all 30.
- **Generation:** `num_predict` 128, temperature 0, seed 42. The router's system prompt (about 300–370 tokens) goes with every request. Outputs were identical across runs: tinyllama averaged 111.9 completion tokens, phi-2 93.2, mistral-7b 107.5.
- **Cold start:** before each model, every model is unloaded. The first call (a fixed warm-up prompt) is reported separately as the cold start: Ollama's `load_duration`, and the client-measured time to first token.
- **Metrics:**
  - **tok/s:** total `eval_count` / total `eval_duration` over the 30 prompts.
  - **TTFT:** client-measured, from sending the request to the first streamed token, with the model already loaded. Consecutive requests share the same system-prompt prefix, which Ollama reuses from its cache, so prompt evaluation is short (Prompt eval p50 column).
  - **E2E:** router time plus the full generation.
  - **Router share:** the sum of router time over the sum of E2E time.
- **Percentiles:** nearest rank. With 30 samples, p99 is the maximum.
- **Repeats:** 3 runs. Results are not merged across runs. The summary gives the range over the 3 runs.
- **Rounding:** throughput is rounded down; latency, overhead and memory are rounded up.

### Results: ranges over 3 runs (Silent mode)

| Model | tok/s | TTFT p50 (warm) | E2E p50 | E2E p90 | E2E p99 (max) | Router p50 | Router share of E2E | Cold load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tinyllama | 174.8–225.0 | 53–56 ms | 616–788 ms | 658–824 ms | 686–852 ms | 1.61–1.84 ms | 0.38–0.48% | 1464–2047 ms |
| phi-2 | 50.5–89.9 | 68–71 ms | 1367–2467 ms | 1622–3014 ms | 2068–3096 ms | 1.66–1.86 ms | 0.20–0.31% | 2483–3368 ms |
| mistral-7b | 16.1–20.7 | 86–156 ms | 6251–8154 ms | 6638–8690 ms | 6702–8771 ms | 1.64–1.74 ms | 0.12–0.17% | 4477–4791 ms |

<details>
<summary>Per-run results</summary>

| Run | Model | tok/s | TTFT p50 | Prompt eval p50 | E2E p50 | E2E p90 | E2E p99 | Router p50 | Router p99 | Router share | Cold load | Cold TTFT | VRAM (NVML delta) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | tinyllama | 225.0 | 53 ms | 15 ms | 616 ms | 658 ms | 686 ms | 1.84 ms | 8.1 ms | 0.38% | 2047 ms | 2123 ms | 0.79 GiB |
| 1 | phi-2 | 89.9 | 68 ms | 22 ms | 1367 ms | 1622 ms | 2068 ms | 1.81 ms | 51.9 ms | 0.31% | 3368 ms | 3450 ms | 2.24 GiB |
| 1 | mistral-7b | 20.7 | 86 ms | 43 ms | 6251 ms | 6638 ms | 6702 ms | 1.74 ms | 44.5 ms | 0.12% | 4791 ms | 4923 ms | 4.76 GiB |
| 2 | tinyllama | 188.3 | 56 ms | 15 ms | 740 ms | 770 ms | 789 ms | 1.65 ms | 44.1 ms | 0.48% | 1479 ms | 1542 ms | 0.79 GiB |
| 2 | phi-2 | 57.8 | 69 ms | 22 ms | 2254 ms | 2411 ms | 2478 ms | 1.66 ms | 25.9 ms | 0.20% | 2723 ms | 2793 ms | 2.24 GiB |
| 2 | mistral-7b | 17.5 | 94 ms | 48 ms | 7230 ms | 8099 ms | 8306 ms | 1.64 ms | 51.4 ms | 0.17% | 4477 ms | 4590 ms | 4.76 GiB |
| 3 | tinyllama | 174.8 | 53 ms | 15 ms | 788 ms | 824 ms | 852 ms | 1.61 ms | 35.1 ms | 0.40% | 1464 ms | 1528 ms | 0.79 GiB |
| 3 | phi-2 | 50.5 | 71 ms | 24 ms | 2467 ms | 3014 ms | 3096 ms | 1.86 ms | 39.6 ms | 0.20% | 2483 ms | 2559 ms | 2.24 GiB |
| 3 | mistral-7b | 16.1 | 156 ms | 117 ms | 8154 ms | 8690 ms | 8771 ms | 1.67 ms | 50.6 ms | 0.13% | 4498 ms | 4620 ms | 4.76 GiB |

VRAM (NVML delta) is the GPU's used memory with only that model loaded, minus the memory used before loading it. It is a little higher than Ollama's `size_vram` because it includes the CUDA context.

</details>

### Routed pass (the router picks the model)

The same 30 prompts sent through the full application (`RoutingApplication.process_query`), with all models unloaded first.

| Run | Requests routed to | E2E p50 | E2E p90 | E2E p99 (max) | Router p50 | Router share | Requests that loaded a model |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | phi-2 21, mistral-7b 5, tinyllama 4 | 2288 ms | 2491 ms | 10160 ms | 1.72 ms | 0.20% | 3 |
| 2 | phi-2 21, mistral-7b 5, tinyllama 4 | 2545 ms | 3143 ms | 10272 ms | 1.78 ms | 0.17% | 3 |
| 3 | phi-2 21, mistral-7b 5, tinyllama 4 | 5224 ms | 5578 ms | 10661 ms | 1.66 ms | 0.13% | 3 |

The 3 loads per run are each model's first use. The p99 (max) request is mistral-7b's first request, which includes its load. All three models fit in VRAM together (7.35 GiB), so no model was evicted during the pass. The prompt labels are not the classifier's output: the router sent 21 of the 30 prompts to phi-2, because phi-2 serves code, general reasoning, conversation, and custom reasoning (by fallback).

### What these numbers do and don't show

- **Throttling.** Throughput fell from run to run: phi-2 from 89.9 to 50.5 tok/s, mistral-7b from 20.7 to 16.1. The prompts, outputs and token counts were identical in every run, which points to the GPU's power and thermal limits, not to the router. Treat these as Silent-mode, throttled numbers, not this GPU's capability.
- **Router overhead.** The router adds about 1.6–1.9 ms at the median, under 0.5% of end-to-end latency in every run. Its p99 reached 25.9–51.9 ms for the larger models; the spikes are rare and not yet explained (probably Python scheduling: GIL contention with the resource-monitor thread is one candidate).
- **Not measured.** Output quality is not measured here. Throughput under concurrent requests is not measured either: the benchmark sends one request at a time.

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
- Ollama and llama.cpp for local inference
- The AI/ML community for inspiration and best practices

---

**Note**: Routing and inference are real, but this is not production-hardened: there is no authentication or rate limiting, the intent classifier is trained on synthetic keyword data, the custom reasoning engines are placeholders, and the cost estimate is a rough formula.
