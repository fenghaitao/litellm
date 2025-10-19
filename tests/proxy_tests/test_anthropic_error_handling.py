"""
Tests for Anthropic error handling module.

This module tests the error response formatting and exception mapping
to ensure all errors are returned in Anthropic's error format.
"""

import pytest
from litellm import exceptions as litellm_exceptions
from litellm.proxy.anthropic_endpoints.error_handling import (
    map_litellm_exception_to_anthropic_error,
    format_anthropic_error_response,
    extract_error_message,
    handle_validation_error,
    handle_provider_error,
    handle_authentication_error,
    handle_rate_limit_error,
    handle_generic_error,
)
from litellm.proxy.anthropic_endpoints.validation import InvalidRequestError


class TestExceptionMapping:
    """Test mapping of LiteLLM exceptions to Anthropic error types."""
    
    def test_authentication_error_mapping(self):
        """Test that AuthenticationError maps to authentication_error."""
        exception = litellm_exceptions.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "authentication_error"
        assert status_code == 401
    
    def test_permission_denied_error_mapping(self):
        """Test that PermissionDeniedError maps to permission_error."""
        import httpx
        exception = litellm_exceptions.PermissionDeniedError(
            message="Access denied",
            llm_provider="openai",
            model="gpt-4",
            response=httpx.Response(status_code=403, request=httpx.Request(method="POST", url="https://api.openai.com"))
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "permission_error"
        assert status_code == 403
    
    def test_not_found_error_mapping(self):
        """Test that NotFoundError maps to not_found_error."""
        exception = litellm_exceptions.NotFoundError(
            message="Model not found",
            model="gpt-5",
            llm_provider="openai"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "not_found_error"
        assert status_code == 404
    
    def test_rate_limit_error_mapping(self):
        """Test that RateLimitError maps to rate_limit_error."""
        exception = litellm_exceptions.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "rate_limit_error"
        assert status_code == 429
    
    def test_bad_request_error_mapping(self):
        """Test that BadRequestError maps to invalid_request_error."""
        exception = litellm_exceptions.BadRequestError(
            message="Invalid parameter",
            model="gpt-4",
            llm_provider="openai"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "invalid_request_error"
        assert status_code == 400
    
    def test_context_window_exceeded_error_mapping(self):
        """Test that ContextWindowExceededError maps to invalid_request_error."""
        exception = litellm_exceptions.ContextWindowExceededError(
            message="Context window exceeded",
            model="gpt-4",
            llm_provider="openai"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "invalid_request_error"
        assert status_code == 400
    
    def test_timeout_error_mapping(self):
        """Test that Timeout maps to api_error."""
        exception = litellm_exceptions.Timeout(
            message="Request timeout",
            model="gpt-4",
            llm_provider="openai"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "api_error"
        assert status_code == 408
    
    def test_service_unavailable_error_mapping(self):
        """Test that ServiceUnavailableError maps to overloaded_error."""
        exception = litellm_exceptions.ServiceUnavailableError(
            message="Service unavailable",
            llm_provider="openai",
            model="gpt-4"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "overloaded_error"
        assert status_code == 503
    
    def test_internal_server_error_mapping(self):
        """Test that InternalServerError maps to api_error."""
        exception = litellm_exceptions.InternalServerError(
            message="Internal server error",
            llm_provider="openai",
            model="gpt-4"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "api_error"
        assert status_code == 500
    
    def test_api_error_with_529_status_mapping(self):
        """Test that APIError with 529 status maps to overloaded_error."""
        exception = litellm_exceptions.APIError(
            status_code=529,
            message="System overloaded",
            llm_provider="openai",
            model="gpt-4"
        )
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "overloaded_error"
        assert status_code == 529
    
    def test_validation_error_mapping(self):
        """Test that InvalidRequestError maps correctly."""
        exception = InvalidRequestError("model is required")
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "invalid_request_error"
        assert status_code == 400
    
    def test_unknown_exception_mapping(self):
        """Test that unknown exceptions map to api_error."""
        exception = ValueError("Something went wrong")
        error_type, status_code = map_litellm_exception_to_anthropic_error(exception)
        assert error_type == "api_error"
        assert status_code == 500


class TestErrorResponseFormatting:
    """Test formatting of error responses in Anthropic format."""
    
    def test_format_authentication_error(self):
        """Test formatting of authentication error."""
        exception = litellm_exceptions.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = format_anthropic_error_response(exception)
        
        assert status_code == 401
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "authentication_error"
        assert "Invalid API key" in error_response["error"]["message"]
    
    def test_format_rate_limit_error(self):
        """Test formatting of rate limit error."""
        exception = litellm_exceptions.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = format_anthropic_error_response(exception)
        
        assert status_code == 429
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in error_response["error"]["message"]
    
    def test_format_validation_error(self):
        """Test formatting of validation error."""
        exception = InvalidRequestError("model is required")
        error_response, status_code = format_anthropic_error_response(exception)
        
        assert status_code == 400
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "invalid_request_error"
        assert error_response["error"]["message"] == "model is required"
    
    def test_format_overloaded_error(self):
        """Test formatting of overloaded error."""
        exception = litellm_exceptions.ServiceUnavailableError(
            message="Service overloaded",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = format_anthropic_error_response(exception)
        
        assert status_code == 503
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "overloaded_error"
        assert "Service overloaded" in error_response["error"]["message"]
    
    def test_format_with_override_error_type(self):
        """Test formatting with overridden error type."""
        exception = Exception("Generic error")
        error_response, status_code = format_anthropic_error_response(
            exception,
            error_type="custom_error",
            status_code=418
        )
        
        assert status_code == 418
        assert error_response["error"]["type"] == "custom_error"


class TestErrorMessageExtraction:
    """Test extraction of error messages from exceptions."""
    
    def test_extract_message_from_litellm_exception(self):
        """Test extracting message from LiteLLM exception."""
        exception = litellm_exceptions.BadRequestError(
            message="Invalid parameter value",
            model="gpt-4",
            llm_provider="openai"
        )
        message = extract_error_message(exception)
        assert "Invalid parameter value" in message
        # Should not have "litellm.BadRequestError:" prefix
        assert not message.startswith("litellm.BadRequestError:")
    
    def test_extract_message_from_validation_error(self):
        """Test extracting message from validation error."""
        exception = InvalidRequestError("messages is required")
        message = extract_error_message(exception)
        assert message == "messages is required"
    
    def test_extract_message_from_generic_exception(self):
        """Test extracting message from generic exception."""
        exception = ValueError("Something went wrong")
        message = extract_error_message(exception)
        assert message == "Something went wrong"


class TestErrorHandlers:
    """Test specific error handler functions."""
    
    def test_handle_validation_error(self):
        """Test validation error handler."""
        exception = InvalidRequestError("model is required")
        error_response, status_code = handle_validation_error(exception)
        
        assert status_code == 400
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "invalid_request_error"
        assert error_response["error"]["message"] == "model is required"
    
    def test_handle_provider_error(self):
        """Test provider error handler."""
        exception = litellm_exceptions.APIError(
            status_code=500,
            message="Provider API error",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = handle_provider_error(exception)
        
        assert status_code == 500
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "api_error"
    
    def test_handle_authentication_error(self):
        """Test authentication error handler."""
        exception = litellm_exceptions.AuthenticationError(
            message="Invalid credentials",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = handle_authentication_error(exception)
        
        assert status_code == 401
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "authentication_error"
    
    def test_handle_rate_limit_error(self):
        """Test rate limit error handler."""
        exception = litellm_exceptions.RateLimitError(
            message="Too many requests",
            llm_provider="openai",
            model="gpt-4"
        )
        error_response, status_code = handle_rate_limit_error(exception)
        
        assert status_code == 429
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "rate_limit_error"
    
    def test_handle_generic_error(self):
        """Test generic error handler."""
        exception = RuntimeError("Unexpected error")
        error_response, status_code = handle_generic_error(exception)
        
        assert status_code == 500
        assert error_response["type"] == "error"
        assert error_response["error"]["type"] == "api_error"
        assert "Unexpected error" in error_response["error"]["message"]


class TestErrorResponseStructure:
    """Test that all error responses follow Anthropic's structure."""
    
    def test_error_response_has_required_fields(self):
        """Test that error responses have all required fields."""
        exception = litellm_exceptions.BadRequestError(
            message="Test error",
            model="gpt-4",
            llm_provider="openai"
        )
        error_response, _ = format_anthropic_error_response(exception)
        
        # Check top-level structure
        assert "type" in error_response
        assert error_response["type"] == "error"
        
        # Check error object structure
        assert "error" in error_response
        assert isinstance(error_response["error"], dict)
        
        # Check error fields
        assert "type" in error_response["error"]
        assert "message" in error_response["error"]
        assert isinstance(error_response["error"]["type"], str)
        assert isinstance(error_response["error"]["message"], str)
    
    def test_all_error_types_are_valid(self):
        """Test that all mapped error types are valid Anthropic error types."""
        valid_error_types = {
            "invalid_request_error",
            "authentication_error",
            "permission_error",
            "not_found_error",
            "rate_limit_error",
            "api_error",
            "overloaded_error",
        }
        
        # Test various exception types
        exceptions = [
            litellm_exceptions.AuthenticationError("test", "openai", "gpt-4"),
            litellm_exceptions.BadRequestError("test", "gpt-4", "openai"),
            litellm_exceptions.RateLimitError("test", "openai", "gpt-4"),
            litellm_exceptions.ServiceUnavailableError("test", "openai", "gpt-4"),
            InvalidRequestError("test"),
        ]
        
        for exception in exceptions:
            error_type, _ = map_litellm_exception_to_anthropic_error(exception)
            assert error_type in valid_error_types, f"Invalid error type: {error_type}"
