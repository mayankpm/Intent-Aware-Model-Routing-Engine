#!/usr/bin/env python3
"""
Demo script for testing the Logic-Based Routing Engine
"""

import requests
import time
import json
from typing import List, Dict

# API base URL
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict = None, params: Dict = None):
    """Test an API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=data, params=params)
        else:
            print(f"❌ Unsupported method: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {BASE_URL}. Is the server running?")
        return None
    except Exception as e:
        print(f"❌ Error testing {endpoint}: {str(e)}")
        return None

def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_result(title: str, result: Dict):
    """Print a result in a formatted way"""
    print(f"\n📋 {title}")
    print("-" * 40)
    print(json.dumps(result, indent=2))

def test_system_status():
    """Test system status endpoint"""
    print_section("System Status")
    
    result = test_endpoint("/status")
    if result:
        print_result("System Status", result)
        return True
    return False

def test_models():
    """Test models endpoint"""
    print_section("Available Models")
    
    result = test_endpoint("/models")
    if result:
        print_result("Available Models", result)
        return True
    return False

def test_intent_classification():
    """Test intent classification with various queries"""
    print_section("Intent Classification")
    
    test_queries = [
        "Write a Python function to calculate fibonacci numbers",
        "Create a story about a robot who learns to paint",
        "Explain the concept of recursion in programming",
        "Apply business logic to optimize the workflow process",
        "Hello, how are you today?",
        "Translate this text to Spanish",
        "Analyze the performance of this algorithm and suggest improvements"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        result = test_endpoint("/classify-intent", method="POST", params={"query": query})
        if result:
            print(f"   Intent: {result.get('intent', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Reasoning: {result.get('reasoning', 'N/A')}")

def test_query_processing():
    """Test query processing with various types of queries"""
    print_section("Query Processing")
    
    test_cases = [
        {
            "query": "Write a function to sort a list in Python",
            "description": "Code Generation"
        },
        {
            "query": "Create a short story about a time traveler",
            "description": "Creative Writing"
        },
        {
            "query": "Explain the concept of machine learning step by step",
            "description": "General Reasoning"
        },
        {
            "query": "Apply domain expertise to optimize the business process workflow",
            "description": "Custom Reasoning"
        },
        {
            "query": "Hello! How are you doing today?",
            "description": "Conversation"
        },
        {
            "query": "Summarize the key points of this document",
            "description": "Specialized Tasks"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"   Query: {test_case['query']}")
        
        # Test with auto custom reasoning enabled
        request_data = {
            "query": test_case['query'],
            "user_id": f"demo_user_{i}",
            "auto_custom_reasoning": True
        }
        
        result = test_endpoint("/query", method="POST", data=request_data)
        if result:
            print(f"   Model Used: {result.get('model_used', 'unknown')}")
            print(f"   Intent: {result.get('intent_classified', 'unknown')}")
            print(f"   Processing Time: {result.get('processing_time', 0):.3f}s")
            
            routing_decision = result.get('routing_decision', {})
            if routing_decision:
                print(f"   Confidence: {routing_decision.get('confidence', 0):.2f}")
                print(f"   Reasoning: {routing_decision.get('reasoning', 'N/A')}")

            metrics = result.get('metrics') or {}
            if metrics:
                tps = metrics.get('tokens_per_second')
                ttft = metrics.get('time_to_first_token_ms')
                print(f"   Backend: {metrics.get('backend')} ({metrics.get('backend_model')})")
                print(f"   Routing: {metrics.get('routing_time_ms', 0):.2f} ms | "
                      f"TTFT: {f'{ttft:.0f} ms' if ttft is not None else 'n/a'} | "
                      f"Tokens/s: {f'{tps:.1f}' if tps else 'n/a'}")
            snippet = (result.get('response') or '')[:100]
            print(f"   Response: {snippet}{'...' if len(result.get('response') or '') > 100 else ''}")

def test_quick_queries():
    """Test quick query endpoint"""
    print_section("Quick Query Tests")
    
    quick_queries = [
        "How do I implement a binary search algorithm?",
        "Tell me a story about a magical forest",
        "What is the difference between supervised and unsupervised learning?",
        "Apply specialized knowledge to analyze this business case",
        "Hi there! Nice to meet you"
    ]
    
    for i, query in enumerate(quick_queries, 1):
        print(f"\n🔍 Quick Test {i}: {query}")
        result = test_endpoint("/test-query", method="POST", params={"query": query})
        if result:
            print(f"   Model: {result.get('model_used', 'unknown')}")
            print(f"   Intent: {result.get('intent_classified', 'unknown')}")
            print(f"   Time: {result.get('processing_time', 0):.3f}s")
            
            # Show a snippet of the response
            response = result.get('response', '')
            if response:
                snippet = response[:100] + "..." if len(response) > 100 else response
                print(f"   Response: {snippet}")

def test_performance_stats():
    """Test performance statistics"""
    print_section("Performance Statistics")
    
    # Get routing stats
    routing_stats = test_endpoint("/routing-stats")
    if routing_stats:
        print_result("Routing Statistics", routing_stats)
    
    # Get performance stats
    perf_stats = test_endpoint("/performance-stats")
    if perf_stats:
        print_result("Performance Statistics", perf_stats)

def run_demo():
    """Run the complete demo"""
    print("🚀 Logic-Based Routing Engine Demo")
    print("=" * 60)
    print("This demo will test various aspects of the routing engine.")
    print("Make sure the server is running at http://localhost:8000")
    print()
    
    # Test if server is running
    print("🔍 Checking if server is running...")
    status = test_endpoint("/")
    if not status:
        print("❌ Server is not running. Please start the server first:")
        print("   python run.py")
        return

    print("✅ Server is running!")
    backend = (test_endpoint("/status") or {}).get("backend", {})
    print(f"🔧 Inference backend: {backend.get('active', 'unknown')} ({backend.get('reason', '')})")
    if backend.get("warning"):
        print(f"⚠️  {backend['warning']}")
    
    # Run all tests
    tests = [
        ("System Status", test_system_status),
        ("Available Models", test_models),
        ("Intent Classification", test_intent_classification),
        ("Query Processing", test_query_processing),
        ("Quick Queries", test_quick_queries),
        ("Performance Statistics", test_performance_stats)
    ]
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} completed")
        except Exception as e:
            print(f"❌ {test_name} failed: {str(e)}")
    
    print("\n" + "="*60)
    print("🎉 Demo completed!")
    print("Check the results above to see how the routing engine works.")
    print("You can also visit http://localhost:8000/docs for API documentation.")

if __name__ == "__main__":
    run_demo() 