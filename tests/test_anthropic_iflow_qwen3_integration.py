#!/usr/bin/env python3
"""
Test script for iflow/Qwen3-Coder integration with enhanced Anthropic endpoints.
Verifies that Claude Code compatibility works with Qwen3-Coder backend.
"""

import json
import os
import requests
import time
from typing import Dict, Any

# Configuration
PROXY_BASE_URL = "http://localhost:4000"
API_KEY = os.getenv('LITELLM_MASTER_KEY', 'sk-litellm-master-20251018-cee48b03eefc0d39e9f690b48c358fe2')  # Use the config master key
IFLOW_MODEL_MAPPED = "claude-3-sonnet-20240229"  # Maps to iflow/Qwen3-Coder via model mapping

def make_request(endpoint: str, data: Dict[str, Any] = None, method: str = "POST") -> requests.Response:
    """Make a request to the enhanced Anthropic proxy."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    url = f"{PROXY_BASE_URL}{endpoint}"
    
    if method == "GET":
        return requests.get(url, headers=headers)
    else:
        return requests.post(url, headers=headers, json=data)

def test_health_check():
    """Test basic health endpoint with authentication."""
    print("🔍 Testing health check...")
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"{PROXY_BASE_URL}/health", headers=headers)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_enhanced_models_endpoint():
    """Test the enhanced /v1/models endpoint."""
    print("\n🧪 Testing enhanced /v1/models endpoint...")
    
    try:
        response = make_request("/v1/models", method="GET")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            print(f"✅ Models endpoint working! Found {len(models)} models")
            
            # Print some example models
            claude_models = [m for m in models if 'claude' in m.get('id', '').lower()]
            print(f"   📋 Claude models available: {len(claude_models)}")
            for model in claude_models[:3]:
                print(f"      - {model.get('id', 'unknown')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False

def test_token_counting():
    """Test enhanced token counting with iflow mapping."""
    print("\n🧪 Testing enhanced token counting...")
    
    test_data = {
        "model": IFLOW_MODEL_MAPPED,
        "messages": [
            {"role": "user", "content": "Write a Python function to sort a list"},
            {"role": "assistant", "content": "I'll help you write a sorting function."},
            {"role": "user", "content": "Make it use quicksort algorithm"}
        ],
        "system": "You are a helpful coding assistant specialized in Python."
    }
    
    try:
        response = make_request("/v1/count_tokens", test_data)
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get('input_tokens', 0)
            print(f"✅ Token counting working! Input tokens: {tokens}")
            return True
        else:
            print(f"❌ Token counting failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Token counting error: {e}")
        return False

def test_context_limit_check():
    """Test context limit checking."""
    print("\n🧪 Testing context limit checking...")
    
    test_data = {
        "model": IFLOW_MODEL_MAPPED,
        "messages": [
            {"role": "user", "content": "Explain Python classes and inheritance"}
        ],
        "max_tokens": 1000
    }
    
    try:
        response = make_request("/v1/check_context_limit", test_data)
        
        if response.status_code == 200:
            data = response.json()
            fits = data.get('fits_context', False)
            input_tokens = data.get('input_tokens', 0)
            max_context = data.get('max_context_tokens', 0)
            print(f"✅ Context limit check working!")
            print(f"   - Fits context: {fits}")
            print(f"   - Input tokens: {input_tokens}")
            print(f"   - Max context tokens: {max_context}")
            return True
        else:
            print(f"❌ Context limit check failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Context limit check error: {e}")
        return False

def test_basic_completion():
    """Test basic completion with Qwen3-Coder via model mapping."""
    print("\n🧪 Testing basic completion (Claude → Qwen3-Coder)...")
    
    test_data = {
        "model": IFLOW_MODEL_MAPPED,
        "messages": [
            {
                "role": "user", 
                "content": "Write a simple Python function to calculate the factorial of a number"
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    try:
        response = make_request("/v1/messages", test_data)
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', [])
            if content and len(content) > 0:
                text_content = content[0].get('text', '')
                print(f"✅ Basic completion working!")
                print(f"   📝 Response preview: {text_content[:100]}...")
                print(f"   🎯 Model used: {data.get('model', 'unknown')}")
                print(f"   📊 Usage: {data.get('usage', {})}")
                return True
            else:
                print(f"❌ Empty response content")
                return False
        else:
            print(f"❌ Basic completion failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Basic completion error: {e}")
        return False

def test_streaming_completion():
    """Test streaming completion."""
    print("\n🧪 Testing streaming completion...")
    
    test_data = {
        "model": IFLOW_MODEL_MAPPED,
        "messages": [
            {"role": "user", "content": "Explain Python decorators in simple terms"}
        ],
        "max_tokens": 300,
        "stream": True
    }
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        
        response = requests.post(
            f"{PROXY_BASE_URL}/v1/messages",
            headers=headers,
            json=test_data,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Streaming connection established")
            
            event_count = 0
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        event_count += 1
                        if event_count <= 5:  # Show first few events
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                event_type = data.get('type', 'unknown')
                                print(f"   📡 Event {event_count}: {event_type}")
                            except json.JSONDecodeError:
                                pass
                        
                        if event_count >= 10:  # Stop after a few events
                            break
            
            print(f"✅ Streaming working! Received {event_count} events")
            return True
        else:
            print(f"❌ Streaming failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        return False

def test_tool_calling():
    """Test tool calling with Qwen3-Coder."""
    print("\n🧪 Testing tool calling...")
    
    test_data = {
        "model": IFLOW_MODEL_MAPPED,
        "messages": [
            {"role": "user", "content": "What's the current weather in San Francisco?"}
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "max_tokens": 500
    }
    
    try:
        response = make_request("/v1/messages", test_data)
        
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', [])
            
            # Check if tool was called
            tool_uses = [block for block in content if block.get('type') == 'tool_use']
            if tool_uses:
                print(f"✅ Tool calling working! Found {len(tool_uses)} tool calls")
                for tool in tool_uses:
                    print(f"   🔧 Tool: {tool.get('name')} with input: {tool.get('input', {})}")
                return True
            else:
                print(f"✅ Tool calling endpoint working (no tools called this time)")
                print(f"   📝 Response: {content[0].get('text', '')[:100] if content else 'Empty'}...")
                return True
        else:
            print(f"❌ Tool calling failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Tool calling error: {e}")
        return False

def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 Starting comprehensive test of iflow/Qwen3-Coder integration")
    print(f"🎯 Testing proxy at: {PROXY_BASE_URL}")
    print(f"🔑 Using API key: {API_KEY[:10] if API_KEY != 'your-proxy-master-key-here' else 'DEFAULT'}...")
    print(f"🤖 Testing model mapping: {IFLOW_MODEL_MAPPED} → iflow/Qwen3-Coder")
    print("=" * 70)
    
    # Wait a moment for proxy to be ready
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health_check),
        ("Enhanced Models Endpoint", test_enhanced_models_endpoint),
        ("Token Counting", test_token_counting),
        ("Context Limit Check", test_context_limit_check),
        ("Basic Completion", test_basic_completion),
        ("Streaming Completion", test_streaming_completion),
        ("Tool Calling", test_tool_calling),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! iflow/Qwen3-Coder integration is working perfectly!")
        print("\n💡 Ready for Claude Code integration:")
        print(f"   - Base URL: {PROXY_BASE_URL}")
        print(f"   - API Key: {API_KEY if len(API_KEY) <= 20 else API_KEY[:10] + '...'}")
        print(f"   - Use any Claude model name (e.g., {IFLOW_MODEL_MAPPED})")
        print(f"   - Environment: LITELLM_MASTER_KEY={os.getenv('LITELLM_MASTER_KEY', 'NOT SET')[:10]}...")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check configuration and try again.")
        print("\n🔍 Troubleshooting tips:")
        print("   - Verify IFLOW_API_KEY is set correctly")
        print("   - Check proxy is running with correct config")
        print("   - Ensure API key in this script matches proxy master_key")

if __name__ == "__main__":
    print("🔧 iflow/Qwen3-Coder Integration Test")
    print("="*50)
    
    # Check environment variables
    iflow_key = os.getenv('IFLOW_API_KEY')
    if not iflow_key:
        print("⚠️  Warning: IFLOW_API_KEY environment variable not set")
        print("   The proxy may fail to connect to iflow")
    else:
        print(f"✅ IFLOW_API_KEY found: {iflow_key[:10]}...")
    
    # Check master key
    master_key = os.getenv('LITELLM_MASTER_KEY')
    if not master_key:
        print("⚠️  Warning: LITELLM_MASTER_KEY environment variable not set")
        print("   Using fallback API key from script")
        if API_KEY == "your-proxy-master-key-here":
            print("   Please set LITELLM_MASTER_KEY environment variable")
            print("   Or update API_KEY in this script")
    else:
        print(f"✅ LITELLM_MASTER_KEY found: {master_key[:10]}...")
        print(f"   Using master key for authentication")
    
    print(f"\n🎯 Starting tests against {PROXY_BASE_URL}...")
    run_comprehensive_test()
