"""
Error handling module for Anthropic Messages API.

This module provides error response formatting and exception mapping
to ensure all errors are returned in Anthropic's error format.
"""

import json
from typing import Any, Dict, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm import exceptions as litellm_exceptions
from litellm.proxy.anthropic_endpoints.validation import InvalidRequestError


def map_litellm_exception_to_anthropic_error(
    exception: Exception,
) -> Tuple[str, int]:
    """
    Map LiteLLM exceptions to Anthropic error types and status codes.
    
    Args:
        exception: The exception to map
        
    Returns:
        Tuple of (error_type, status_code)
    """
    # Authentication errors
    if isinstance(exception, litellm_exceptions.AuthenticationError):
        return "authentication_error", 401
    
    # Permission errors
    if isinstance(exception, litellm_exceptions.PermissionDeniedError):
        return "permission_error", 403
    
    # Not found errors
    if isinstance(exception, litellm_exceptions.NotFoundError):
        return "not_found_error", 404
    
    # Rate limit errors
    if isinstance(exception, litellm_exceptions.RateLimitError):
        return "rate_limit_error", 429
    
    # Bad request errors (including validation)
    if isinstance(exception, (
        litellm_exceptions.BadRequestError,
        litellm_exceptions.ContextWindowExceededError,
        litellm_exceptions.ContentPolicyViolationError,
        litellm_exceptions.RejectedRequestError,
        litellm_exceptions.UnsupportedParamsError,
    )):
        return "invalid_request_error", 400
    
    # Unprocessable entity errors
    if isinstance(exception, litellm_exceptions.UnprocessableEntityError):
        return "invalid_request_error", 422
    
    # Timeout errors
    if isinstance(exception, litellm_exceptions.Timeout):
        return "api_error", 408
    
    # Service unavailable errors
    if isinstance(exception, (
        litellm_exceptions.ServiceUnavailableError,
        litellm_exceptions.MidStreamFallbackError,
    )):
        return "overloaded_error", 503
    
    # Internal server errors
    if isinstance(exception, litellm_exceptions.InternalServerError):
        return "api_error", 500
    
    # API connection errors
    if isinstance(exception, litellm_exceptions.APIConnectionError):
        return "api_error", 500
    
    # API response validation errors
    if isinstance(exception, litellm_exceptions.APIResponseValidationError):
        return "api_error", 500
    
    # Generic API errors
    if isinstance(exception, litellm_exceptions.APIError):
        status_code = getattr(exception, "status_code", 500)
        if status_code == 529:
            return "overloaded_error", 529
        elif status_code >= 500:
            return "api_error", status_code
        elif status_code == 429:
            return "rate_limit_error", 429
        elif status_code == 401:
            return "authentication_error", 401
        elif status_code == 403:
            return "permission_error", 403
        elif status_code == 404:
            return "not_found_error", 404
        else:
            return "invalid_request_error", status_code
    
    # Validation errors from our module
    if isinstance(exception, InvalidRequestError):
        return exception.error_type, exception.status_code
    
    # Default to api_error for unknown exceptions
    return "api_error", 500


def format_anthropic_error_response(
    exception: Exception,
    error_type: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """
    Format an exception as an Anthropic-compatible error response.
    
    Args:
        exception: The exception to format
        error_type: Optional override for error type
        status_code: Optional override for status code
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    # Map exception to error type and status code if not provided
    if error_type is None or status_code is None:
        mapped_error_type, mapped_status_code = map_litellm_exception_to_anthropic_error(exception)
        error_type = error_type or mapped_error_type
        status_code = status_code or mapped_status_code
    
    # Extract error message
    error_message = extract_error_message(exception)
    
    # Log the error
    verbose_proxy_logger.error(
        f"Anthropic endpoint error: type={error_type}, status={status_code}, "
        f"exception={type(exception).__name__}, message={error_message}"
    )
    
    # Build Anthropic error response
    error_response = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": error_message
        }
    }
    
    return error_response, status_code


def extract_error_message(exception: Exception) -> str:
    """
    Extract a clean error message from an exception.
    
    Args:
        exception: The exception to extract message from
        
    Returns:
        Clean error message string
    """
    # Try to get message attribute first
    if hasattr(exception, "message"):
        message = exception.message
        # Remove "litellm." prefix from error messages
        if isinstance(message, str) and message.startswith("litellm."):
            # Extract just the message part after the exception class name
            parts = message.split(": ", 1)
            if len(parts) > 1:
                return parts[1]
        return str(message)
    
    # Fall back to string representation
    message = str(exception)
    
    # Remove "litellm." prefix if present
    if message.startswith("litellm."):
        parts = message.split(": ", 1)
        if len(parts) > 1:
            return parts[1]
    
    return message


def create_error_response_json(
    error_type: str,
    message: str,
    status_code: int = 400,
) -> Tuple[str, int]:
    """
    Create a JSON error response string.
    
    Args:
        error_type: Anthropic error type
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (json_string, status_code)
    """
    error_response = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        }
    }
    
    return json.dumps(error_response), status_code


def handle_validation_error(
    validation_error: InvalidRequestError,
) -> Tuple[Dict[str, Any], int]:
    """
    Handle validation errors specifically.
    
    Args:
        validation_error: The validation error to handle
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    verbose_proxy_logger.warning(
        f"Anthropic request validation failed: {validation_error.message}"
    )
    
    return validation_error.to_dict(), validation_error.status_code


def handle_provider_error(
    provider_exception: Exception,
) -> Tuple[Dict[str, Any], int]:
    """
    Handle errors from LLM providers.
    
    Args:
        provider_exception: The provider exception to handle
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    error_type, status_code = map_litellm_exception_to_anthropic_error(provider_exception)
    
    verbose_proxy_logger.error(
        f"Provider error: type={error_type}, status={status_code}, "
        f"exception={type(provider_exception).__name__}"
    )
    
    return format_anthropic_error_response(
        provider_exception,
        error_type=error_type,
        status_code=status_code,
    )


def handle_authentication_error(
    auth_exception: Exception,
) -> Tuple[Dict[str, Any], int]:
    """
    Handle authentication errors.
    
    Args:
        auth_exception: The authentication exception to handle
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    error_message = extract_error_message(auth_exception)
    
    verbose_proxy_logger.warning(
        f"Authentication error: {error_message}"
    )
    
    error_response = {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": error_message
        }
    }
    
    return error_response, 401


def handle_rate_limit_error(
    rate_limit_exception: Exception,
) -> Tuple[Dict[str, Any], int]:
    """
    Handle rate limit errors.
    
    Args:
        rate_limit_exception: The rate limit exception to handle
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    error_message = extract_error_message(rate_limit_exception)
    
    verbose_proxy_logger.warning(
        f"Rate limit exceeded: {error_message}"
    )
    
    error_response = {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": error_message
        }
    }
    
    return error_response, 429


def handle_generic_error(
    exception: Exception,
) -> Tuple[Dict[str, Any], int]:
    """
    Handle generic/unknown errors.
    
    Args:
        exception: The exception to handle
        
    Returns:
        Tuple of (error_response_dict, status_code)
    """
    error_message = extract_error_message(exception)
    
    verbose_proxy_logger.exception(
        f"Unexpected error in Anthropic endpoint: {error_message}"
    )
    
    error_response = {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": error_message
        }
    }
    
    return error_response, 500
