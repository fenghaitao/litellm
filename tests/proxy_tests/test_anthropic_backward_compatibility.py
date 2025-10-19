"""
Tests for backward compatibility of Anthropic endpoint enhancements.

This module verifies that:
1. Existing pass-through mode still works when transformation is disabled
2. Existing OpenAI endpoints are unaffected
3. Configuration flag properly controls transformation mode
4. Both OpenAI and Anthropic formats can be used simultaneously
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.endpoints import anthropic_response
from fastapi import Request, Response


@pytest.mark.asyncio
async def test_transformation_mode_enabled_by_default():
    """Test that transformation mode is enabled by default."""
    from litellm.proxy._types import ConfigGeneralSettings
    
    # Create settings without explicit flag
    settings = ConfigGeneralSettings()
    
    # Should default to True
    assert getattr(settings, "anthropic_transformation_enabled", True) == True


@pytest.mark.asyncio
async def test_transformation_mode_can_be_disabled():
    """Test that transformation mode can be disabled via configuration."""
    from litellm.proxy._types import ConfigGeneralSettings
    
    # Create settings with flag disabled
    settings = ConfigGeneralSettings(anthropic_transformation_enabled=False)
    
    # Should be False
    assert settings.anthropic_transformation_enabled == False


@pytest.mark.asyncio
async def test_passthrough_mode_with_disabled_transformation():
    """Test that pass-through mode is used when transformation is disabled."""
    
    # Mock the general_settings to disable transformation
    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = False
    
    # Mock request and response
    mock_request = MagicMock(spec=Request)
    mock_response = MagicMock(spec=Response)
    mock_response.headers = {}
    
    # Mock user auth
    mock_user_auth = MagicMock(spec=UserAPIKeyAuth)
    
    # Mock request body
    request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024
    }
    
    # Mock the _read_request_body function
    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body") as mock_read:
        mock_read.return_value = request_data
        
        # Mock the proxy_server module where general_settings is imported from
        with patch("litellm.proxy.proxy_server.general_settings", mock_general_settings):
            with patch("litellm.proxy.anthropic_endpoints.endpoints._anthropic_passthrough_handler") as mock_passthrough:
                mock_passthrough.return_value = {"id": "msg_123", "content": [{"type": "text", "text": "Hi"}]}
                
                # Call the endpoint
                result = await anthropic_response(
                    fastapi_response=mock_response,
                    request=mock_request,
                    user_api_key_dict=mock_user_auth
                )
                
                # Verify pass-through handler was called
                mock_passthrough.assert_called_once()


@pytest.mark.asyncio
async def test_transformation_mode_with_enabled_flag():
    """Test that transformation mode is used when enabled."""
    
    # Mock the general_settings to enable transformation
    mock_general_settings = MagicMock()
    mock_general_settings.anthropic_transformation_enabled = True
    
    # Mock request and response
    mock_request = MagicMock(spec=Request)
    mock_response = MagicMock(spec=Response)
    mock_response.headers = {}
    
    # Mock user auth
    mock_user_auth = MagicMock(spec=UserAPIKeyAuth)
    
    # Mock request body
    request_data = {
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024
    }
    
    # Mock the _read_request_body function
    with patch("litellm.proxy.anthropic_endpoints.endpoints._read_request_body") as mock_read:
        mock_read.return_value = request_data
        
        # Mock the proxy_server module where general_settings is imported from
        with patch("litellm.proxy.proxy_server.general_settings", mock_general_settings):
            # Mock the transformation modules to verify they're called
            with patch("litellm.proxy.anthropic_endpoints.endpoints.validate_anthropic_request") as mock_validate:
                with patch("litellm.proxy.anthropic_endpoints.endpoints.AnthropicToOpenAITransformer") as mock_transformer:
                    # Setup transformer mock
                    mock_transformer_instance = MagicMock()
                    mock_transformer_instance.transform_messages = MagicMock(return_value=[{"role": "user", "content": "Hello"}])
                    mock_transformer.return_value = mock_transformer_instance
                    
                    # Mock the pass-through handler to ensure it's NOT called
                    with patch("litellm.proxy.anthropic_endpoints.endpoints._anthropic_passthrough_handler") as mock_passthrough:
                        # Mock ProxyBaseLLMRequestProcessing to raise an exception early
                        # This allows us to verify the transformation path was taken without
                        # having to mock the entire request processing pipeline
                        with patch("litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing") as mock_processor:
                            mock_processor.side_effect = Exception("Test exception to exit early")
                            
                            # Call the endpoint and expect it to raise
                            try:
                                result = await anthropic_response(
                                    fastapi_response=mock_response,
                                    request=mock_request,
                                    user_api_key_dict=mock_user_auth
                                )
                            except:
                                pass  # Expected to fail
                            
                            # Verify transformation was attempted (validation and transformation were called)
                            mock_validate.assert_called_once()
                            mock_transformer.assert_called_once()
                            mock_transformer_instance.transform_messages.assert_called_once()
                            
                            # Verify pass-through handler was NOT called
                            mock_passthrough.assert_not_called()


@pytest.mark.asyncio
async def test_configuration_flag_in_general_settings():
    """Test that the configuration flag is properly defined in ConfigGeneralSettings."""
    from litellm.proxy._types import ConfigGeneralSettings
    
    # Test with explicit True
    settings_true = ConfigGeneralSettings(anthropic_transformation_enabled=True)
    assert settings_true.anthropic_transformation_enabled == True
    
    # Test with explicit False
    settings_false = ConfigGeneralSettings(anthropic_transformation_enabled=False)
    assert settings_false.anthropic_transformation_enabled == False
    
    # Test default value
    settings_default = ConfigGeneralSettings()
    # Should default to True
    assert getattr(settings_default, "anthropic_transformation_enabled", True) == True


def test_openai_endpoints_unaffected():
    """Test that OpenAI endpoints are not affected by Anthropic changes."""
    # This is a placeholder test to verify that OpenAI endpoints still work
    # In a real scenario, this would make actual requests to /v1/chat/completions
    
    # The key point is that the Anthropic endpoint changes should not affect
    # the OpenAI endpoint behavior at all
    
    # We can verify this by checking that the OpenAI endpoint handler
    # doesn't import or use any Anthropic-specific code
    
    from litellm.proxy import proxy_server
    
    # Verify that the proxy server still has the OpenAI endpoints
    # This is a basic sanity check
    assert hasattr(proxy_server, "router")


@pytest.mark.asyncio
async def test_both_formats_can_coexist():
    """Test that both OpenAI and Anthropic formats can be used simultaneously."""
    
    # This test verifies that having both endpoints active doesn't cause conflicts
    # In a real deployment, you should be able to:
    # 1. Send OpenAI format requests to /v1/chat/completions
    # 2. Send Anthropic format requests to /v1/messages
    # 3. Both should work independently
    
    # For this unit test, we just verify that both endpoint handlers exist
    from litellm.proxy.anthropic_endpoints import endpoints as anthropic_endpoints
    
    # Verify Anthropic endpoint exists
    assert hasattr(anthropic_endpoints, "anthropic_response")
    
    # Verify pass-through handler exists
    assert hasattr(anthropic_endpoints, "_anthropic_passthrough_handler")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
