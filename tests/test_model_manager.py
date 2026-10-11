"""ModelManager lifecycle on the mock backend."""
import threading

from conftest import make_app
from src.models import IntentType


def test_unload_model_unloads_the_named_model(mock_app):
    mm = mock_app.model_manager
    mm._load_model("phi-2")
    mm._load_model("tinyllama")
    # phi-2 is least recently used; unloading tinyllama must not unload phi-2
    assert mm.unload_model("tinyllama")
    assert mm.get_loaded_models() == ["phi-2"]


def test_reload_model_does_not_deadlock(mock_app):
    mm = mock_app.model_manager
    mm._load_model("phi-2")
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", mm.reload_model("phi-2")))
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "reload_model deadlocked"
    assert result["ok"]


def test_lru_eviction(mock_app):
    mm = mock_app.model_manager
    mm.max_cache_size = 2
    for name in ("phi-2", "tinyllama", "mistral-7b"):
        mm._load_model(name)
    assert set(mm.get_loaded_models()) == {"tinyllama", "mistral-7b"}


def test_unavailable_model_is_not_loaded(llm_only_app):
    mm = llm_only_app.model_manager
    assert mm._load_model("llama-2-13b") is None
    assert "unavailable" in mm.last_error
    assert mm.process_query("hi", IntentType.CONVERSATION, "business-logic-engine") is None


def test_process_query_records_routing_time(mock_app):
    resp = mock_app.model_manager.process_query("hi", IntentType.CONVERSATION, "phi-2",
                                                routing_time=0.002)
    assert resp.metrics.routing_time_ms == 2.0
    assert resp.metrics.total_latency_ms >= 2.0
    assert resp.metrics.model_load_ms == 0.0  # first use loaded it (mock load is instant)
    again = mock_app.model_manager.process_query("hi", IntentType.CONVERSATION, "phi-2")
    assert again.metrics.model_load_ms is None  # already loaded
