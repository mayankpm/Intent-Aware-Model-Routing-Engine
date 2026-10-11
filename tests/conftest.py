import os
import sys
from pathlib import Path

import pytest

# src.main builds a global app at import time; keep it on the mock backend
os.environ.setdefault("ROUTER_BACKEND", "mock")
os.environ.setdefault("ROUTER_POLICY", "legacy")  # learned-policy tests opt in explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends import BackendSelection, MockBackend  # noqa: E402
from src.main import RoutingApplication  # noqa: E402
from src.models import IntentType  # noqa: E402
from src.resources import ResourceMonitor  # noqa: E402


def make_resources(free_vram_gb=8.0, available_ram_gb=16.0, cpu_cores=16,
                   loaded=None, gpu=True):
    return {
        "available_ram_gb": available_ram_gb,
        "total_ram_gb": 16.0,
        "cpu_usage_percent": 10.0,
        "available_cpu_cores": cpu_cores,
        "disk_usage_percent": 50.0,
        "gpu": {"name": "test-gpu", "total_vram_gb": 8.0, "free_vram_gb": free_vram_gb,
                "used_vram_gb": 8.0 - free_vram_gb, "source": "test"} if gpu else None,
        "backend_loaded_vram_gb": dict(loaded or {}),
    }


class FixedResourceMonitor(ResourceMonitor):
    """Returns a fixed snapshot so routing tests don't depend on this machine's load."""

    def __init__(self, snapshot):
        self.backend = None
        self.ttl = 0
        self._fixed = snapshot

    def snapshot(self, refresh=False):
        return self._fixed


def make_app(backend=None, resources=None):
    backend = backend or MockBackend()
    selection = BackendSelection("mock", "mock", "test", backend)
    return RoutingApplication(backend_selection=selection,
                              resource_monitor=FixedResourceMonitor(resources or make_resources()))


# One query per intent. Each is checked against the classifier in test_routing.py.
INTENT_QUERIES = {
    IntentType.CODE_GENERATION: "Write a Python function to reverse a linked list",
    IntentType.CREATIVE_WRITING: "Write a short story about a robot who learns to paint",
    IntentType.GENERAL_REASONING: "Explain why the sky is blue",
    IntentType.CUSTOM_REASONING: "Apply business logic to optimize the workflow process",
    IntentType.CONVERSATION: "Hello, how are you today?",
    IntentType.SPECIALIZED_TASKS: "Translate this sentence to Spanish: the cat is on the table",
}


@pytest.fixture
def mock_app():
    return make_app()


@pytest.fixture
def llm_only_app():
    """Mock backend that only runs the LLMs Ollama has, mirroring the Ollama setup."""
    return make_app(MockBackend(run_non_llm=False))
