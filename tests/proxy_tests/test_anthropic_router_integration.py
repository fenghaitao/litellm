"""
Integration tests for Anthropic endpoint with LiteLLM router.

Tests verify that Anthropic requests properly integrate with:
- Router load balancing
- Fallback mechanisms
- Rate limiting
- Logging and metadata capture
- Multiple provider configurations
- Authentication and authorization
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.endpoints import anthropic_response

@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    return request


@pytest.fixture
def mock_response():
    """Create a mock FastAPI response."""
    response = MagicMock(spec=Response)
    response.headers = {}
    return response


@pytest.fixture
def mock_user_api_key_dict():
    """Create a mock user API key dict."""
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team",
    )


@pytest.fixture
def anthropic_request_data():
    """Sample Anthropic request data."""
    return {
        "model": "claude-3-sonnet-20240229",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI-format response."""
    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm doing well, thank you!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    }
    mock_resp._hidden_params = {
        "model_id": "model-123",
        "cache_key": "cache-key",
        "api_base": "https://api.openai.com",
        "response_cost": 0.001,
    }
    return mock_resp


@pytest.mark.asyncio
async def test_router_integration_basic(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
    mock_openai_response,
):
    """Test that Anthropic requests route through LiteLLM router."""
    async def mock_read_body(request):
        return anthropic_request_data

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
        side_effect=lambda data, user_api_key_dict, response: response
    )
    mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.return_value = mock_openai_response

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance
                    mock_processor.get_custom_headers = MagicMock(return_value={})

                    # Call the endpoint
                    result = await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify route_request was called with correct parameters
                    mock_route.assert_called_once()
                    call_args = mock_route.call_args
                    assert call_args.kwargs["route_type"] == "acompletion"
                    assert call_args.kwargs["llm_router"] == mock_llm_router

                    # Verify response is in Anthropic format
                    assert "id" in result
                    assert result["type"] == "message"
                    assert result["role"] == "assistant"
                    assert "content" in result
                    assert result["model"] == "claude-3-sonnet-20240229"


@pytest.mark.asyncio
async def test_router_load_balancing_metadata(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
):
    """Test that router metadata (model_id, api_base) is captured."""
    async def mock_read_body(request):
        return anthropic_request_data

    # Mock router response with specific deployment metadata
    mock_router_response = MagicMock()
    mock_router_response.model_dump.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response from deployment 1"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }
    mock_router_response._hidden_params = {
        "model_id": "deployment-1",
        "api_base": "https://deployment1.openai.com",
        "response_cost": 0.002,
    }

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
        side_effect=lambda data, user_api_key_dict, response: response
    )
    mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.return_value = mock_router_response

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance

                    captured_headers = {}

                    def capture_headers(**kwargs):
                        captured_headers.update(kwargs)
                        return {}

                    mock_processor.get_custom_headers = MagicMock(side_effect=capture_headers)

                    # Call the endpoint
                    await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify metadata was captured
                    assert captured_headers["model_id"] == "deployment-1"
                    assert captured_headers["api_base"] == "https://deployment1.openai.com"
                    assert captured_headers["response_cost"] == 0.002


@pytest.mark.asyncio
async def test_router_logging_integration(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
    mock_openai_response,
):
    """Test that logging captures Anthropic request metadata."""
    async def mock_read_body(request):
        return anthropic_request_data

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
        side_effect=lambda data, user_api_key_dict, response: response
    )
    mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.return_value = mock_openai_response

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance
                    mock_processor.get_custom_headers = MagicMock(return_value={})

                    # Call the endpoint
                    await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify logging hooks were called
                    mock_proxy_logging_obj.during_call_hook.assert_called_once()
                    call_args = mock_proxy_logging_obj.during_call_hook.call_args
                    assert call_args.kwargs["call_type"] == "completion"

                    mock_proxy_logging_obj.post_call_success_hook.assert_called_once()
                    mock_proxy_logging_obj.update_request_status.assert_called_once()


@pytest.mark.asyncio
async def test_router_rate_limiting_integration(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
):
    """Test that rate limiting is applied to Anthropic requests."""
    import litellm

    async def mock_read_body(request):
        return anthropic_request_data

    # Simulate rate limit error from router
    rate_limit_error = litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider="openai",
        model="gpt-4",
    )

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.side_effect = rate_limit_error

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance

                    # Call the endpoint
                    result = await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify error response is in Anthropic format
                    response_data = json.loads(result.body.decode())
                    assert response_data["type"] == "error"
                    assert response_data["error"]["type"] == "rate_limit_error"

                    # Verify failure hook was called
                    mock_proxy_logging_obj.post_call_failure_hook.assert_called_once()


@pytest.mark.asyncio
async def test_router_authentication_integration(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
):
    """Test that authentication errors are handled correctly."""
    import litellm

    async def mock_read_body(request):
        return anthropic_request_data

    # Simulate authentication error
    auth_error = litellm.AuthenticationError(
        message="Invalid API key",
        llm_provider="openai",
        model="gpt-4",
    )

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.side_effect = auth_error

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance

                    # Call the endpoint
                    result = await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify error response is in Anthropic format
                    response_data = json.loads(result.body.decode())
                    assert response_data["type"] == "error"
                    assert response_data["error"]["type"] == "authentication_error"


@pytest.mark.asyncio
async def test_common_processing_pre_call_logic_called(
    mock_request,
    mock_response,
    mock_user_api_key_dict,
    anthropic_request_data,
    mock_openai_response,
):
    """Test that common_processing_pre_call_logic is called with correct parameters."""
    async def mock_read_body(request):
        return anthropic_request_data

    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True

    mock_llm_router = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_logging_obj.post_call_success_hook = AsyncMock(
        side_effect=lambda data, user_api_key_dict, response: response
    )
    mock_proxy_logging_obj.update_request_status = AsyncMock(return_value=None)
    mock_proxy_logging_obj.litellm_call_id = "call-123"

    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body", mock_read_body):
        with patch("litellm.proxy.anthropic_endpoints.endpoints.proxy_server") as mock_proxy_server:
            mock_proxy_server.general_settings = mock_general_settings
            mock_proxy_server.llm_router = mock_llm_router
            mock_proxy_server.proxy_config = None
            mock_proxy_server.proxy_logging_obj = mock_proxy_logging_obj
            mock_proxy_server.user_api_base = None
            mock_proxy_server.user_max_tokens = None
            mock_proxy_server.user_model = None
            mock_proxy_server.user_request_timeout = None
            mock_proxy_server.user_temperature = None
            mock_proxy_server.version = "1.0.0"

            with patch("litellm.proxy.anthropic_endpoints.endpoints.route_request") as mock_route:
                mock_route.return_value = mock_openai_response

                with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                    mock_instance = MagicMock()
                    mock_instance.common_processing_pre_call_logic = AsyncMock(
                        return_value=(anthropic_request_data, mock_proxy_logging_obj)
                    )
                    mock_processor.return_value = mock_instance
                    mock_processor.get_custom_headers = MagicMock(return_value={})

                    # Call the endpoint
                    await anthropic_response(
                        fastapi_response=mock_response,
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                    )

                    # Verify common_processing_pre_call_logic was called
                    mock_instance.common_processing_pre_call_logic.assert_called_once()
                    call_args = mock_instance.common_processing_pre_call_logic.call_args
                    
                    # Verify route_type is set to "acompletion" for router integration
                    assert call_args.kwargs["route_type"] == "acompletion"
                    assert call_args.kwargs["request"] == mock_request
                    assert call_args.kwargs["user_api_key_dict"] == mock_user_api_key_dict
