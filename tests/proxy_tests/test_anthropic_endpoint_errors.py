"""
Integration tests for Anthropic endpoint error handling.

This module tests that the /v1/messages endpoint properly handles
and formats errors in Anthropic's error format.
"""

import json
import pytest
from litellm import exceptions as litellm_exceptions
from litellm.proxy.anthropic_endpoints.validation import InvalidRequestError


class TestErrorTypeMapping:
    """Test that different error types are properly mapped."""
    
    def test_authentication_error_format(self):
        """Test authentication error formatting."""
        from litellm.proxy.anthropic_endpoints.error_handling import handle_authentication_error
        
        exception = litellm_exceptions.AuthenticationError(
            message="Invalid API key provided",
            llm_provider="openai",
            model="gpt-4"
        )
        
        error_response, status_code = handle_authentication_error(exception)
        
        assert status_code == 401
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "authentication_error"
        assert "Invalid API key" in error_response["error"]["message"]
    
    def test_rate_limit_error_format(self):
        """Test rate limit error formatting."""
        from litellm.proxy.anthropic_endpoints.error_handling import handle_rate_limit_error
        
        exception = litellm_exceptions.RateLimitError(
            message="Rate limit exceeded. Please try again later.",
            llm_provider="openai",
            model="gpt-4"
        )
        
        error_response, status_code = handle_rate_limit_error(exception)
        
        assert status_code == 429
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in error_response["error"]["message"]
    
    def test_provider_error_format(self):
        """Test provider error formatting."""
        from litellm.proxy.anthropic_endpoints.error_handling import handle_provider_error
        
        exception = litellm_exceptions.ServiceUnavailableError(
            message="The service is temporarily unavailable",
            llm_provider="openai",
            model="gpt-4"
        )
        
        error_response, status_code = handle_provider_error(exception)
        
        assert status_code == 503
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "overloaded_error"
        assert "temporarily unavailable" in error_response["error"]["message"]
    
    def test_generic_error_format(self):
        """Test generic error formatting."""
        from litellm.proxy.anthropic_endpoints.error_handling import handle_generic_error
        
        exception = RuntimeError("Unexpected internal error")
        
        error_response, status_code = handle_generic_error(exception)
        
        assert status_code == 500
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "api_error"
        assert "Unexpected internal error" in error_response["error"]["message"]


class TestErrorResponseStructure:
    """Test that all error responses follow Anthropic's structure."""
    
    def test_all_errors_have_required_fields(self):
        """Test that all error types have required fields."""
        from litellm.proxy.anthropic_endpoints.error_handling import (
            handle_validation_error,
            handle_authentication_error,
            handle_rate_limit_error,
            handle_provider_error,
            handle_generic_error,
        )
        
        # Test different error handlers
        handlers_and_exceptions = [
            (handle_validation_error, InvalidRequestError("test")),
            (handle_authentication_error, litellm_exceptions.AuthenticationError("test", "openai", "gpt-4")),
            (handle_rate_limit_error, litellm_exceptions.RateLimitError("test", "openai", "gpt-4")),
            (handle_provider_error, litellm_exceptions.APIError(500, "test", "openai", "gpt-4")),
            (handle_generic_error, Exception("test")),
        ]
        
        for handler, exception in handlers_and_exceptions:
            error_response, status_code = handler(exception)
            
            # Verify structure
            assert "type" in error_response
            assert error_response["type"] == "error"
            assert "error" in error_response
            assert "type" in error_response["error"]
            assert "message" in error_response["error"]
            assert isinstance(error_response["error"]["type"], str)
            assert isinstance(error_response["error"]["message"], str)
            assert isinstance(status_code, int)
            assert 400 <= status_code < 600
