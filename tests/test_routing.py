"""Routing tests on the mock backend with fixed resources."""
import pytest

from conftest import INTENT_QUERIES, make_app, make_resources
from src.backends import MockBackend
from src.models import IntentType

# Mock backend that runs every registry entry, including the custom reasoning placeholders
EXPECTED_ALL_ENGINES = {
    IntentType.CODE_GENERATION: "phi-2",
    IntentType.CREATIVE_WRITING: "tinyllama",
    IntentType.GENERAL_REASONING: "business-logic-engine",
    IntentType.CUSTOM_REASONING: "business-logic-engine",
    IntentType.CONVERSATION: "phi-2",
    IntentType.SPECIALIZED_TASKS: "domain-expert-engine",
}

# Only the LLMs that Ollama has pulled (tinyllama, phi, mistral): the Ollama routing table
EXPECTED_LLM_ONLY = {
    IntentType.CODE_GENERATION: "phi-2",
    IntentType.CREATIVE_WRITING: "tinyllama",
    IntentType.GENERAL_REASONING: "phi-2",
    IntentType.CUSTOM_REASONING: "phi-2",       # no LLM has this capability: fallback
    IntentType.CONVERSATION: "phi-2",
    IntentType.SPECIALIZED_TASKS: "mistral-7b",
}

DISABLED = {"llama-2-7b", "llama-2-13b"}


@pytest.mark.parametrize("intent,query", list(INTENT_QUERIES.items()), ids=lambda v: getattr(v, "value", "q"))
def test_classifier_labels_test_queries(mock_app, intent, query):
    assert mock_app.intent_classifier.classify_intent(query).intent == intent


@pytest.mark.parametrize("intent", list(EXPECTED_ALL_ENGINES), ids=lambda i: i.value)
def test_routing_with_all_engines(mock_app, intent):
    decision = mock_app.routing_engine.route_query(INTENT_QUERIES[intent])
    assert decision.intent == intent
    assert decision.selected_model == EXPECTED_ALL_ENGINES[intent]


@pytest.mark.parametrize("intent", list(EXPECTED_LLM_ONLY), ids=lambda i: i.value)
def test_routing_llm_only_matches_ollama_setup(llm_only_app, intent):
    decision = llm_only_app.routing_engine.route_query(INTENT_QUERIES[intent])
    assert decision.selected_model == EXPECTED_LLM_ONLY[intent]


def test_fallback_returns_available_model(llm_only_app):
    decision = llm_only_app.routing_engine.route_query(INTENT_QUERIES[IntentType.CUSTOM_REASONING])
    info = llm_only_app.model_registry.get_model_info(decision.selected_model)
    assert info is not None and info.is_available and info.is_llm
    assert "fell back" in decision.reasoning
    assert decision.fallback_model is not None
    assert llm_only_app.model_registry.get_model_info(decision.fallback_model).is_available


def test_registry_fallback_prefers_intent_then_general(llm_only_app):
    registry = llm_only_app.model_registry
    assert registry.get_fallback_model(IntentType.CREATIVE_WRITING).name == "tinyllama"
    fallback = registry.get_fallback_model(IntentType.CUSTOM_REASONING)
    assert fallback.is_available and IntentType.GENERAL_REASONING in fallback.capabilities


@pytest.mark.parametrize("query", list(INTENT_QUERIES.values()))
def test_unavailable_models_never_selected(query):
    # Backend that can only run tinyllama: every route must land on it
    app = make_app(MockBackend(runnable=["tinyllama"]))
    decision = app.routing_engine.route_query(query)
    assert decision.selected_model == "tinyllama"
    assert decision.fallback_model in (None, "tinyllama")


@pytest.mark.parametrize("query", list(INTENT_QUERIES.values()))
def test_disabled_models_never_selected(mock_app, query):
    decision = mock_app.routing_engine.route_query(query)
    assert decision.selected_model not in DISABLED
    assert decision.fallback_model not in DISABLED


def test_disabled_models_report_reason(mock_app):
    for name in DISABLED:
        info = mock_app.model_registry.get_model_info(name)
        assert not info.is_available
        assert "not installed" in info.unavailable_reason


def test_non_llm_engines_unavailable_without_mock_support(llm_only_app):
    for name in ("business-logic-engine", "domain-expert-engine"):
        info = llm_only_app.model_registry.get_model_info(name)
        assert not info.is_llm and not info.is_available


def test_low_vram_excludes_large_models():
    # 1 GB free, nothing loaded: only tinyllama (0.65 GB) fits
    app = make_app(MockBackend(run_non_llm=False), make_resources(free_vram_gb=1.0))
    decision = app.routing_engine.route_query(INTENT_QUERIES[IntentType.SPECIALIZED_TASKS])
    assert decision.selected_model == "tinyllama"
    assert app.model_registry.get_model_info("mistral-7b").unavailable_reason.startswith("insufficient")


def test_loaded_model_vram_counts_as_reclaimable():
    # Mistral already holds 4.61 GB; it fits, and phi-2 fits if the backend evicts it
    resources = make_resources(free_vram_gb=0.5, loaded={"mistral:latest": 4.61})
    app = make_app(MockBackend(run_non_llm=False), resources)
    assert app.model_registry.get_model_info("mistral-7b").is_available
    assert app.model_registry.get_model_info("phi-2").is_available


def test_no_gpu_uses_ram_requirement():
    resources = make_resources(gpu=False, available_ram_gb=3.0)
    app = make_app(MockBackend(run_non_llm=False), resources)
    assert app.model_registry.get_model_info("phi-2").is_available        # 2.09 GB
    assert not app.model_registry.get_model_info("mistral-7b").is_available  # 4.61 GB


def test_no_model_available_reports_it():
    app = make_app(MockBackend(runnable=[]))
    decision = app.routing_engine.route_query("Hello there")
    assert decision.selected_model == "fallback"
    assert "No model is available" in decision.reasoning
