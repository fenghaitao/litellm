"""
Unit tests for IFLOW configuration.

These tests validate the IFlowChatConfig class which extends OpenAIGPTConfig.
IFLOW is an OpenAI-compatible provider with minor customizations.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion
from litellm.llms.iflow.chat.transformation import IFlowChatConfig


class TestIFlowConfig:
    """Test class for IFLOW functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = IFlowChatConfig()
        headers = {}
        api_key = "fake-iflow-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="Qwen3-Coder",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

        # We can't directly test the api_base value here since validate_environment
        # only returns the headers, but we can verify it doesn't raise an exception
        # which would happen if api_base handling was incorrect

    def test_get_openai_compatible_provider_info(self):
        """Test the provider info method returns correct defaults"""
        config = IFlowChatConfig()
        
        # Test with no parameters
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://apis.iflow.cn/v1"
        # api_key might be None or set from environment - both are valid
        
        # Test with custom api_base
        custom_base = "https://custom.iflow.api.com/v1"
        api_base, api_key = config._get_openai_compatible_provider_info(custom_base, None)
        assert api_base == custom_base
        
        # Test with custom api_key
        custom_key = "custom-key-123"
        api_base, api_key = config._get_openai_compatible_provider_info(None, custom_key)
        assert api_key == custom_key

    def test_get_complete_url(self):
        """Test URL completion functionality"""
        config = IFlowChatConfig()
        
        # Test with default base
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="Qwen3-Coder",
            optional_params={},
            litellm_params={},
            stream=False
        )
        assert url == "https://apis.iflow.cn/v1/chat/completions"
        
        # Test with custom base (without trailing /chat/completions)
        custom_base = "https://custom.api.com/v2"
        url = config.get_complete_url(
            api_base=custom_base,
            api_key=None,
            model="Qwen3-Coder",
            optional_params={},
            litellm_params={},
            stream=False
        )
        assert url == "https://custom.api.com/v2/chat/completions"
        
        # Test with custom base (with trailing /chat/completions)
        custom_base_complete = "https://custom.api.com/v2/chat/completions"
        url = config.get_complete_url(
            api_base=custom_base_complete,
            api_key=None,
            model="Qwen3-Coder",
            optional_params={},
            litellm_params={},
            stream=False
        )
        assert url == "https://custom.api.com/v2/chat/completions"

    def test_message_transformation(self):
        """Test message transformation handles content list to string conversion"""
        config = IFlowChatConfig()
        
        # Test messages with content as list (should be converted to string)
        messages_with_list = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }
        ]
        
        # This should not raise an exception and should handle the conversion
        transformed = config._transform_messages(messages_with_list, "Qwen3-Coder", is_async=False)
        assert len(transformed) == 1
        assert transformed[0]["role"] == "user"
        # The content should be converted to string format

    @pytest.mark.skip(reason="Mock test - requires respx setup")
    def test_iflow_completion_mock(self, respx_mock):
        """
        Mock test for IFLOW completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-iflow-key"
        api_base = "https://apis.iflow.cn/v1"
        model = "iflow/Qwen3-Coder"
        model_name = "Qwen3-Coder"  # The actual model name without provider prefix

        # Mock the HTTP request to the iflow API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```\n\nThis is a simple recursive implementation of the Fibonacci sequence.',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 25,
                    "total_tokens": 37,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "```python" in response.choices[0].message.content
        assert "fibonacci" in response.choices[0].message.content.lower()

    @pytest.mark.skip(reason="Mock test - requires respx setup")
    def test_iflow_completion_mock_with_tools(self, respx_mock):
        """
        Mock test for IFLOW completion with function calling.
        Tests the tool/function calling capabilities.
        """

        litellm.disable_aiohttp_transport = True

        # Set up environment variables for the test
        api_key = "fake-iflow-key"
        api_base = "https://apis.iflow.cn/v1"
        model = "iflow/Qwen3-Coder"
        model_name = "Qwen3-Coder"

        # Mock the HTTP request to the iflow API with tool calls
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Beijing"}'
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 45,
                    "completion_tokens": 15,
                    "total_tokens": 60,
                },
            },
            status_code=200,
        )

        # Define tools for the test
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like in Beijing?"}
            ],
            tools=tools,
            tool_choice="auto",
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "tool_calls")
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) > 0

        # Check tool call details
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        assert "Beijing" in tool_call.function.arguments

    @pytest.mark.skip(reason="Mock test - requires respx setup")
    def test_iflow_streaming_mock(self, respx_mock):
        """
        Mock test for IFLOW streaming completion.
        """

        litellm.disable_aiohttp_transport = True

        # Set up environment variables for the test
        api_key = "fake-iflow-key"
        api_base = "https://apis.iflow.cn/v1"
        model = "iflow/Qwen3-Coder"

        # Mock streaming response
        streaming_response = [
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{"content":"Here"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"Qwen3-Coder","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n'
        ]

        respx_mock.post(f"{api_base}/chat/completions").respond(
            content="".join(streaming_response),
            headers={"content-type": "text/plain"},
            status_code=200,
        )

        # Make streaming completion call
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            stream=True,
            api_key=api_key,
            api_base=api_base,
        )

        # Collect streaming chunks
        chunks = []
        for chunk in response:
            chunks.append(chunk)

        # Verify we got streaming chunks
        assert len(chunks) > 0
        
        # Verify chunk structure
        for chunk in chunks:
            assert hasattr(chunk, "choices")
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                assert isinstance(chunk.choices[0].delta.content, str)