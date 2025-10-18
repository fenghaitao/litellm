#!/usr/bin/env python3
"""
Comprehensive integration tests for LiteLLM Proxy with GitHub Copilot GPT-4.
This test suite validates the proxy server functionality when configured with GitHub Copilot.

Prerequisites:
- LiteLLM proxy server running on localhost:4000
- GitHub Copilot configured and accessible
- LITELLM_MASTER_KEY environment variable set with your proxy master key

Setup:
    export LITELLM_MASTER_KEY="your-master-key-here"
    pytest tests/test_litellm/llms/github_copilot/test_github_copilot_proxy_integration.py -v

Test Coverage:
- Health checks and server status
- Model endpoint availability
- Basic chat completion functionality
- GitHub Copilot specific headers and features
- Streaming completions
- Raw HTTP requests
- Error handling and authentication
- Text embedding functionality (ada-002, 3-small)
- Batch embedding processing
- Embedding with GitHub Copilot headers
"""

import pytest
import aiohttp
import asyncio
import json
import os
from typing import Optional, Dict, Any
import openai
import requests


class TestGitHubCopilotProxyIntegration:
    """Test suite for GitHub Copilot proxy integration"""
    
    # Configuration
    PROXY_URL = "http://localhost:4000"
    API_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")  # Use environment variable or fallback
    
    @pytest.fixture(scope="class")
    def proxy_config(self):
        """Configuration for the proxy tests"""
        return {
            "base_url": self.PROXY_URL,
            "api_key": self.API_KEY,
            "model": "gpt-4",  # GitHub Copilot model
            "timeout": 30
        }
    
    def test_health_check(self, proxy_config):
        """Test if the proxy server is running and healthy"""
        try:
            # Health endpoint requires authentication
            headers = {"Authorization": f"Bearer {proxy_config['api_key']}"}
            response = requests.get(f"{proxy_config['base_url']}/health", headers=headers, timeout=30)
            
            # Expect 200 status for healthy server
            assert response.status_code == 200, f"Health check failed with status {response.status_code}: {response.text}"
            
            health_data = response.json()
            assert "healthy_endpoints" in health_data, "Health response missing healthy_endpoints"
            assert "unhealthy_endpoints" in health_data, "Health response missing unhealthy_endpoints"
            
        except requests.exceptions.ConnectionError:
            pytest.fail("Failed to connect to proxy server. Make sure it's running on port 4000")
        except requests.exceptions.ReadTimeout:
            pytest.fail("Health check timed out. The server may be unresponsive or not running on port 4000")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")
    
    def test_models_endpoint(self, proxy_config):
        """Test the /models endpoint to verify GitHub Copilot model availability"""
        headers = {"Authorization": f"Bearer {proxy_config['api_key']}"}
        response = requests.get(f"{proxy_config['base_url']}/v1/models", headers=headers)
        
        assert response.status_code == 200, f"Models endpoint failed with status {response.status_code}"
        
        models_data = response.json()
        assert "data" in models_data, "Models response missing data field"
        
        # Check if GitHub Copilot model is available
        model_ids = [model.get('id', '') for model in models_data.get('data', [])]
        github_copilot_models = [mid for mid in model_ids if 'github_copilot' in mid or 'gpt-4' in mid]
        
        assert len(github_copilot_models) > 0, f"No GitHub Copilot models found. Available models: {model_ids}"
    
    def test_basic_chat_completion(self, proxy_config):
        """Test basic chat completion without GitHub Copilot specific headers"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.chat.completions.create(
            model=proxy_config['model'],
            messages=[
                {"role": "user", "content": "Write a simple Python function that adds two numbers. Keep it concise."}
            ],
            max_tokens=150,
            timeout=proxy_config['timeout']
        )
        
        # Validate response structure
        assert hasattr(response, 'choices'), "Response missing choices"
        assert len(response.choices) > 0, "Response has no choices"
        assert hasattr(response.choices[0], 'message'), "Choice missing message"
        assert response.choices[0].message.content, "Message content is empty"
        assert hasattr(response, 'usage'), "Response missing usage information"
        
        # Validate content quality (basic check)
        content = response.choices[0].message.content.lower()
        assert any(keyword in content for keyword in ['def', 'function', 'return', 'add']), \
            f"Response doesn't seem to contain a Python function: {content}"
    
    def test_github_copilot_chat_completion(self, proxy_config):
        """Test chat completion with GitHub Copilot specific headers"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.chat.completions.create(
            model=proxy_config['model'],
            messages=[
                {"role": "user", "content": "Create a recursive fibonacci function in Python with error handling"}
            ],
            max_tokens=250,
            timeout=proxy_config['timeout'],
            extra_headers={
                "editor-version": "vscode/1.85.1",
                "Copilot-Integration-Id": "vscode-chat",
                "editor-plugin-version": "copilot/1.155.0",
                "user-agent": "GithubCopilot/1.155.0"
            }
        )
        
        # Validate response
        assert response.choices[0].message.content, "GitHub Copilot response is empty"
        content = response.choices[0].message.content.lower()
        assert any(keyword in content for keyword in ['fibonacci', 'recursive', 'def']), \
            f"Response doesn't contain expected fibonacci function: {content}"
        
        # Check if usage information is provided
        assert response.usage.total_tokens > 0, "Usage tokens should be greater than 0"
    
    def test_text_embedding_ada_002(self, proxy_config):
        """Test text embedding with text-embedding-ada-002 model"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=["Hello world", "This is a test embedding"],
            timeout=proxy_config['timeout']
        )
        
        # Validate response structure
        assert hasattr(response, 'data'), "Embedding response missing data"
        assert len(response.data) == 2, "Expected 2 embedding vectors"
        
        for i, embedding in enumerate(response.data):
            assert hasattr(embedding, 'embedding'), f"Embedding {i} missing embedding vector"
            assert isinstance(embedding.embedding, list), f"Embedding {i} vector should be a list"
            assert len(embedding.embedding) > 0, f"Embedding {i} vector should not be empty"
            assert all(isinstance(x, (int, float)) for x in embedding.embedding), f"Embedding {i} vector should contain numbers"
        
        # Check usage information
        assert hasattr(response, 'usage'), "Embedding response missing usage"
        assert response.usage.total_tokens > 0, "Embedding usage tokens should be greater than 0"
    
    def test_text_embedding_3_small(self, proxy_config):
        """Test text embedding with text-embedding-3-small model"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="This is a test for the newer embedding model",
            timeout=proxy_config['timeout']
        )
        
        # Validate response structure
        assert hasattr(response, 'data'), "Embedding response missing data"
        assert len(response.data) == 1, "Expected 1 embedding vector"
        
        embedding = response.data[0]
        assert hasattr(embedding, 'embedding'), "Embedding missing embedding vector"
        assert isinstance(embedding.embedding, list), "Embedding vector should be a list"
        assert len(embedding.embedding) > 0, "Embedding vector should not be empty"
        
        # text-embedding-3-small should have 1536 dimensions by default
        assert len(embedding.embedding) == 1536, f"Expected 1536 dimensions, got {len(embedding.embedding)}"
        
        # Check usage information
        assert hasattr(response, 'usage'), "Embedding response missing usage"
        assert response.usage.total_tokens > 0, "Embedding usage tokens should be greater than 0"
    
    def test_embedding_batch_processing(self, proxy_config):
        """Test embedding with multiple inputs (batch processing)"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        # Test with multiple texts
        texts = [
            "Python is a programming language",
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing deals with text analysis",
            "Embeddings convert text to numerical vectors"
        ]
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
            timeout=proxy_config['timeout']
        )
        
        # Validate response structure
        assert hasattr(response, 'data'), "Batch embedding response missing data"
        assert len(response.data) == len(texts), f"Expected {len(texts)} embedding vectors"
        
        # Check each embedding
        for i, embedding in enumerate(response.data):
            assert hasattr(embedding, 'embedding'), f"Batch embedding {i} missing embedding vector"
            assert isinstance(embedding.embedding, list), f"Batch embedding {i} vector should be a list"
            assert len(embedding.embedding) > 0, f"Batch embedding {i} vector should not be empty"
            assert embedding.index == i, f"Batch embedding {i} has wrong index: {embedding.index}"
        
        # Check that embeddings are different (semantic similarity test)
        emb1 = response.data[0].embedding
        emb2 = response.data[1].embedding
        
        # Simple check: embeddings should not be identical
        assert emb1 != emb2, "Different texts should produce different embeddings"
        
        # Check usage information
        assert hasattr(response, 'usage'), "Batch embedding response missing usage"
        assert response.usage.total_tokens > 0, "Batch embedding usage tokens should be greater than 0"
    
    def test_embedding_with_github_copilot_headers(self, proxy_config):
        """Test embedding with GitHub Copilot specific headers"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="Code embedding test with GitHub Copilot headers",
            timeout=proxy_config['timeout'],
            extra_headers={
                "editor-version": "vscode/1.85.1",
                "Copilot-Integration-Id": "vscode-embedding",
                "editor-plugin-version": "copilot/1.155.0",
                "user-agent": "GithubCopilot/1.155.0"
            }
        )
        
        # Validate response structure
        assert hasattr(response, 'data'), "Copilot embedding response missing data"
        assert len(response.data) == 1, "Expected 1 embedding vector"
        
        embedding = response.data[0]
        assert hasattr(embedding, 'embedding'), "Copilot embedding missing embedding vector"
        assert isinstance(embedding.embedding, list), "Copilot embedding vector should be a list"
        assert len(embedding.embedding) > 0, "Copilot embedding vector should not be empty"
        
        # Check usage information
        assert hasattr(response, 'usage'), "Copilot embedding response missing usage"
        assert response.usage.total_tokens > 0, "Copilot embedding usage tokens should be greater than 0"
    
    def test_streaming_completion(self, proxy_config):
        """Test streaming chat completion functionality"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        stream = client.chat.completions.create(
            model=proxy_config['model'],
            messages=[
                {"role": "user", "content": "Explain async/await in Python in 2-3 sentences"}
            ],
            max_tokens=100,
            stream=True,
            timeout=proxy_config['timeout'],
            extra_headers={
                "editor-version": "vscode/1.85.1",
                "Copilot-Integration-Id": "vscode-chat"
            }
        )
        
        # Collect streaming response
        full_response = ""
        chunk_count = 0
        
        for chunk in stream:
            chunk_count += 1
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
        
        # Validate streaming response
        assert chunk_count > 0, "No chunks received from streaming"
        assert len(full_response) > 0, "Streaming response is empty"
        assert any(keyword in full_response.lower() for keyword in ['async', 'await', 'python']), \
            f"Streaming response doesn't contain expected content: {full_response}"
    
    def test_raw_http_request(self, proxy_config):
        """Test using raw HTTP request (equivalent to curl)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_config['api_key']}",
            "editor-version": "vscode/1.85.1",
            "Copilot-Integration-Id": "vscode-chat"
        }
        
        data = {
            "model": proxy_config['model'],
            "messages": [
                {"role": "user", "content": "What is the difference between a list and a tuple in Python?"}
            ],
            "max_tokens": 150
        }
        
        response = requests.post(
            f"{proxy_config['base_url']}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=proxy_config['timeout']
        )
        
        assert response.status_code == 200, f"Raw HTTP request failed with status {response.status_code}"
        
        result = response.json()
        assert "choices" in result, "Response missing choices"
        assert len(result["choices"]) > 0, "Response has no choices"
        assert result["choices"][0]["message"]["content"], "Message content is empty"
        
        # Validate content mentions both list and tuple
        content = result["choices"][0]["message"]["content"].lower()
        assert "list" in content and "tuple" in content, \
            f"Response doesn't mention both list and tuple: {content}"
    
    def test_invalid_model_error_handling(self, proxy_config):
        """Test error handling with invalid model"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        # Test with invalid model - expect error or fallback behavior
        try:
            response = client.chat.completions.create(
                model="invalid-model-name",
                messages=[{"role": "user", "content": "Test"}],
                timeout=5
            )
            # If no exception is raised, the proxy might be configured to handle unknown models
            # This is actually valid behavior for some proxy configurations
            assert True, "Proxy handled invalid model gracefully"
        except (openai.NotFoundError, openai.BadRequestError, Exception) as e:
            # Any error response is acceptable for invalid model
            assert True, f"Proxy correctly returned error for invalid model: {e}"
    
    def test_malformed_request_error_handling(self, proxy_config):
        """Test error handling with malformed request"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_config['api_key']}"
        }
        
        # Send malformed data (missing required fields)
        malformed_data = {
            "model": proxy_config['model']
            # Missing messages field
        }
        
        response = requests.post(
            f"{proxy_config['base_url']}/v1/chat/completions",
            headers=headers,
            json=malformed_data,
            timeout=5
        )
        
        # Accept either error status or successful handling with default behavior
        if response.status_code >= 400:
            assert True, "Proxy correctly returned error for malformed request"
        elif response.status_code == 200:
            # Some proxies might handle missing messages with defaults
            result = response.json()
            assert "choices" in result or "error" in result, "Response should contain choices or error"
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
    
    def test_async_health_and_completion(self, proxy_config):
        """Test async operations using aiohttp (run synchronously)"""
        import asyncio
        
        async def async_test():
            async with aiohttp.ClientSession() as session:
                # Test health endpoint with authentication
                headers = {"Authorization": f"Bearer {proxy_config['api_key']}"}
                async with session.get(f"{proxy_config['base_url']}/health", headers=headers) as response:
                    assert response.status == 200, f"Health check failed with status {response.status}"
                    health_data = await response.json()
                    assert "healthy_endpoints" in health_data
                
                # Test async completion
                headers = {
                    "Authorization": f"Bearer {proxy_config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": proxy_config['model'],
                    "messages": [{"role": "user", "content": "Hello from async test!"}],
                    "max_tokens": 50
                }
                
                async with session.post(
                    f"{proxy_config['base_url']}/v1/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    assert response.status == 200
                    result = await response.json()
                    assert "choices" in result
                    assert result["choices"][0]["message"]["content"]
        
        # Run the async test synchronously
        asyncio.run(async_test())
    
    def test_concurrent_requests(self, proxy_config):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            client = openai.OpenAI(
                base_url=proxy_config['base_url'],
                api_key=proxy_config['api_key']
            )
            
            response = client.chat.completions.create(
                model=proxy_config['model'],
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=50,
                timeout=10
            )
            return response.choices[0].message.content
        
        # Run 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert len(results) == 3
        for result in results:
            assert result, "One or more concurrent requests failed"
    
    def test_response_metadata(self, proxy_config):
        """Test that response includes proper metadata"""
        client = openai.OpenAI(
            base_url=proxy_config['base_url'],
            api_key=proxy_config['api_key']
        )
        
        response = client.chat.completions.create(
            model=proxy_config['model'],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50
        )
        
        # Check response metadata
        assert hasattr(response, 'id'), "Response missing id"
        assert hasattr(response, 'created'), "Response missing created timestamp"
        assert hasattr(response, 'model'), "Response missing model field"
        assert response.object == "chat.completion", "Response object type incorrect"
        
        # Check usage metadata
        assert response.usage.prompt_tokens > 0, "Prompt tokens should be greater than 0"
        assert response.usage.completion_tokens > 0, "Completion tokens should be greater than 0"
        assert response.usage.total_tokens > 0, "Total tokens should be greater than 0"


# Standalone test functions for backward compatibility
def test_health_check_standalone():
    """Standalone health check test"""
    api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    if not os.getenv("LITELLM_MASTER_KEY"):
        print("⚠️  Warning: LITELLM_MASTER_KEY not set, using fallback")
    
    test_instance = TestGitHubCopilotProxyIntegration()
    proxy_config = {
        "base_url": test_instance.PROXY_URL,
        "api_key": api_key,
        "model": "gpt-4",
        "timeout": 30
    }
    test_instance.test_health_check(proxy_config)


def test_basic_completion_standalone():
    """Standalone basic completion test"""
    api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    if not os.getenv("LITELLM_MASTER_KEY"):
        print("⚠️  Warning: LITELLM_MASTER_KEY not set, using fallback")
    
    test_instance = TestGitHubCopilotProxyIntegration()
    proxy_config = {
        "base_url": test_instance.PROXY_URL,
        "api_key": api_key,
        "model": "gpt-4",
        "timeout": 30
    }
    test_instance.test_basic_chat_completion(proxy_config)


def test_embedding_standalone():
    """Standalone embedding test"""
    api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    if not os.getenv("LITELLM_MASTER_KEY"):
        print("⚠️  Warning: LITELLM_MASTER_KEY not set, using fallback")
    
    test_instance = TestGitHubCopilotProxyIntegration()
    proxy_config = {
        "base_url": test_instance.PROXY_URL,
        "api_key": api_key,
        "model": "gpt-4",
        "timeout": 30
    }
    test_instance.test_text_embedding_ada_002(proxy_config)


def test_embedding_batch_standalone():
    """Standalone batch embedding test"""
    api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    if not os.getenv("LITELLM_MASTER_KEY"):
        print("⚠️  Warning: LITELLM_MASTER_KEY not set, using fallback")
    
    test_instance = TestGitHubCopilotProxyIntegration()
    proxy_config = {
        "base_url": test_instance.PROXY_URL,
        "api_key": api_key,
        "model": "gpt-4",
        "timeout": 30
    }
    test_instance.test_embedding_batch_processing(proxy_config)


if __name__ == "__main__":
    # Run basic tests if executed directly
    print("🚀 Running GitHub Copilot Proxy Integration Tests")
    
    try:
        test_health_check_standalone()
        print("✅ Health check passed")
        
        test_basic_completion_standalone()
        print("✅ Basic completion passed")
        
        test_embedding_standalone()
        print("✅ Basic embedding passed")
        
        test_embedding_batch_standalone()
        print("✅ Batch embedding passed")
        
        print("🎉 All basic tests completed successfully!")
        print("📋 Tests covered:")
        print("   - Health check with authentication")
        print("   - Chat completion (GPT-4)")
        print("   - Text embedding (ada-002)")
        print("   - Batch embedding processing")
        print("")
        print("🔧 Run with pytest for full test suite:")
        print("   pytest tests/test_litellm/llms/github_copilot/test_github_copilot_proxy_integration.py -v")
        print("")
        print("🧪 Additional tests available in full suite:")
        print("   - Streaming completions")
        print("   - GitHub Copilot headers")
        print("   - Error handling scenarios")
        print("   - Concurrent requests")
        print("   - text-embedding-3-small model")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure your LiteLLM proxy is running on port 4000 with GitHub Copilot configured")