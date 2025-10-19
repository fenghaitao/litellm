"""
Unified /v1/messages endpoint - (Anthropic Spec)
"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    create_streaming_response,
)
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.proxy.route_llm_request import route_request
from litellm.types.utils import TokenCountResponse

# Import transformation, validation, and error handling modules
from litellm.proxy.anthropic_endpoints.validation import (
    validate_anthropic_request,
    InvalidRequestError,
)
from litellm.proxy.anthropic_endpoints.transformation import (
    AnthropicToOpenAITransformer,
    OpenAIToAnthropicTransformer,
)
from litellm.proxy.anthropic_endpoints.streaming import AnthropicStreamingHandler
from litellm.proxy.anthropic_endpoints.error_handling import (
    format_anthropic_error_response,
    handle_validation_error,
    handle_provider_error,
    handle_authentication_error,
    handle_rate_limit_error,
    handle_generic_error,
    map_litellm_exception_to_anthropic_error,
)

router = APIRouter()


async def _anthropic_passthrough_handler(
    fastapi_response: Response,
    request: Request,
    data: dict,
    user_api_key_dict: UserAPIKeyAuth,
):
    """
    Pass-through handler for Anthropic Messages API.
    
    This handler routes requests directly to the Anthropic API without transformation,
    maintaining backward compatibility with existing pass-through implementations.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    
    try:
        # Use the existing anthropic_messages handler for pass-through
        # This maintains backward compatibility with the original implementation
        
        # Setup base processing
        base_llm_processor = ProxyBaseLLMRequestProcessing(data=data)
        
        # Apply common pre-call processing
        data, logging_obj = await base_llm_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="anthropic_messages",
            version=version,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
        )
        
        # Setup parallel tasks
        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type="anthropic_messages",
            )
        )
        
        # Route through LiteLLM's anthropic_messages handler
        llm_call = await route_request(
            data=data,
            route_type="anthropic_messages",
            llm_router=llm_router,
            user_model=user_model,
        )
        tasks.append(llm_call)
        
        # Wait for completion
        llm_responses = asyncio.gather(*tasks)
        responses = await llm_responses
        response = responses[1]
        
        # Extract metadata
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        
        # Update request status
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )
        
        # Set custom headers
        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
                hidden_params=hidden_params,
            )
        )
        
        # Handle streaming vs non-streaming
        if data.get("stream", False):
            return await create_streaming_response(
                generator=response,
                media_type="text/event-stream",
                headers=dict(fastapi_response.headers),
            )
        else:
            # Call post-success hooks
            response = await proxy_logging_obj.post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=response,  # type: ignore
            )
            return response
            
    except Exception as e:
        # Handle errors
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=data,
        )
        raise


@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Anthropic Messages API endpoint with optional transformation to OpenAI format.
    
    This endpoint supports two modes:
    
    1. Transformation Mode (default, anthropic_transformation_enabled=true):
       - Accepts Anthropic-formatted requests
       - Transforms them to OpenAI format
       - Routes through LiteLLM's infrastructure to any provider
       - Transforms responses back to Anthropic format
       - Supports: Text/image content, tool calling, streaming, all LiteLLM routing features
    
    2. Pass-Through Mode (anthropic_transformation_enabled=false):
       - Accepts Anthropic-formatted requests
       - Routes directly to Anthropic API without transformation
       - Returns Anthropic-formatted responses
       - Maintains backward compatibility with existing implementations
    
    Configuration:
    Set `anthropic_transformation_enabled: false` in general_settings to use pass-through mode.
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    request_data = await _read_request_body(request=request)
    data: dict = {**request_data}
    
    # Check if transformation mode is enabled (default: True)
    transformation_enabled = True
    if general_settings is not None:
        transformation_enabled = getattr(
            general_settings, 
            "anthropic_transformation_enabled", 
            True
        )
    
    # If transformation is disabled, use pass-through mode
    if not transformation_enabled:
        return await _anthropic_passthrough_handler(
            fastapi_response=fastapi_response,
            request=request,
            data=data,
            user_api_key_dict=user_api_key_dict,
        )
    
    try:
        # Step 1: Validate Anthropic request
        try:
            validate_anthropic_request(data)
        except InvalidRequestError as e:
            error_response, status_code = handle_validation_error(e)
            return Response(
                content=json.dumps(error_response),
                status_code=status_code,
                media_type="application/json"
            )
        
        # Store original model name and system for response transformation
        original_model = data.get("model", "")
        anthropic_system = data.get("system")
        
        # Step 2: Transform Anthropic request to OpenAI format
        anthropic_to_openai = AnthropicToOpenAITransformer()
        
        # Extract Anthropic-specific fields
        anthropic_messages = data.get("messages", [])
        anthropic_tools = data.get("tools")
        anthropic_tool_choice = data.get("tool_choice")
        
        # Transform messages
        openai_messages = anthropic_to_openai.transform_messages(
            anthropic_messages,
            system=anthropic_system
        )
        data["messages"] = openai_messages
        
        # Transform tools if provided
        if anthropic_tools:
            openai_tools = anthropic_to_openai.transform_tools(anthropic_tools)
            data["tools"] = openai_tools
        
        # Transform tool_choice if provided
        if anthropic_tool_choice:
            openai_tool_choice = anthropic_to_openai.transform_tool_choice(
                anthropic_tool_choice
            )
            data["tool_choice"] = openai_tool_choice
        
        # Remove Anthropic-specific fields that don't exist in OpenAI format
        data.pop("system", None)  # Already converted to system message
        
        # Step 3: Use ProxyBaseLLMRequestProcessing for standard processing
        base_llm_processor = ProxyBaseLLMRequestProcessing(data=data)
        
        # Apply common pre-call processing (auth, rate limiting, logging setup)
        data, logging_obj = await base_llm_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            route_type="acompletion",
            version=version,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
        )
        
        # Step 4: Setup parallel tasks (hooks + LLM call)
        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type="completion",
            )
        )
        
        # Step 5: Route the request through LiteLLM's router infrastructure
        # This handles load balancing, fallbacks, rate limiting, etc.
        llm_call = await route_request(
            data=data,
            route_type="acompletion",
            llm_router=llm_router,
            user_model=user_model,
        )
        tasks.append(llm_call)
        
        # Wait for all tasks to complete
        llm_responses = asyncio.gather(*tasks)
        responses = await llm_responses
        response = responses[1]
        
        # Extract metadata from response
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        
        # Update request status for monitoring
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )
        
        verbose_proxy_logger.debug("final response: %s", response)
        
        # Set custom headers with routing metadata
        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
                hidden_params=hidden_params,
            )
        )
        
        # Step 6: Transform response back to Anthropic format
        if data.get("stream", False):
            # Handle streaming responses
            streaming_handler = AnthropicStreamingHandler()
            
            # Transform OpenAI stream to Anthropic SSE format
            anthropic_stream = streaming_handler.transform_stream(
                openai_stream=response,
                model=original_model,
                system=anthropic_system
            )
            
            return await create_streaming_response(
                generator=anthropic_stream,
                media_type="text/event-stream",
                headers=dict(fastapi_response.headers),
            )
        else:
            # Handle non-streaming responses
            openai_to_anthropic = OpenAIToAnthropicTransformer()
            
            # Convert response to dict if needed
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif hasattr(response, "dict"):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"choices": [{"message": {"content": str(response)}}]}
            
            # Transform to Anthropic format
            anthropic_response = openai_to_anthropic.transform_response(
                response_dict,
                original_model=original_model
            )
            
            # Call post-success hooks
            anthropic_response = await proxy_logging_obj.post_call_success_hook(
                data=data, 
                user_api_key_dict=user_api_key_dict, 
                response=anthropic_response  # type: ignore
            )
            
            verbose_proxy_logger.debug("\nResponse from Litellm:\n{}".format(anthropic_response))
            return anthropic_response
            
    except InvalidRequestError as e:
        # Handle validation errors with Anthropic format
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        error_response, status_code = handle_validation_error(e)
        return Response(
            content=json.dumps(error_response),
            status_code=status_code,
            media_type="application/json"
        )
    except litellm.AuthenticationError as e:
        # Handle authentication errors
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        error_response, status_code = handle_authentication_error(e)
        return Response(
            content=json.dumps(error_response),
            status_code=status_code,
            media_type="application/json"
        )
    except litellm.RateLimitError as e:
        # Handle rate limit errors
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        error_response, status_code = handle_rate_limit_error(e)
        return Response(
            content=json.dumps(error_response),
            status_code=status_code,
            media_type="application/json"
        )
    except (
        litellm.BadRequestError,
        litellm.NotFoundError,
        litellm.Timeout,
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
        litellm.APIError,
        litellm.APIConnectionError,
    ) as e:
        # Handle provider errors with proper Anthropic error types
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        error_response, status_code = handle_provider_error(e)
        return Response(
            content=json.dumps(error_response),
            status_code=status_code,
            media_type="application/json"
        )
    except Exception as e:
        # Handle all other errors
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        error_response, status_code = handle_generic_error(e)
        return Response(
            content=json.dumps(error_response),
            status_code=status_code,
            media_type="application/json"
        )


@router.post(
    "/v1/messages/count_tokens",
    tags=["[beta] Anthropic Messages Token Counting"],
    dependencies=[Depends(user_api_key_auth)],
)
async def count_tokens(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
):
    """
    Count tokens for Anthropic Messages API format.
    
    This endpoint follows the Anthropic Messages API token counting specification.
    It accepts the same parameters as the /v1/messages endpoint but returns
    token counts instead of generating a response.
    
    Example usage:
    ```
    curl -X POST "http://localhost:4000/v1/messages/count_tokens?beta=true" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
      -d '{
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}]
      }'
    ```
    
    Returns: {"input_tokens": <number>}
    """
    from litellm.proxy.proxy_server import token_counter as internal_token_counter
    
    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}
        
        # Extract required fields
        model_name = data.get("model")
        messages = data.get("messages", [])
        
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail={"error": "model parameter is required"}
            )
        
        if not messages:
            raise HTTPException(
                status_code=400,
                detail={"error": "messages parameter is required"}
            )
        
        # Create TokenCountRequest for the internal endpoint
        from litellm.proxy._types import TokenCountRequest
        
        token_request = TokenCountRequest(
            model=model_name,
            messages=messages
        )
        
        # Call the internal token counter function with direct request flag set to False
        token_response = await internal_token_counter(
            request=token_request,
            call_endpoint=True,
        )
        _token_response_dict: dict = {}
        if isinstance(token_response, TokenCountResponse):
            _token_response_dict = token_response.model_dump()
        elif isinstance(token_response, dict):
            _token_response_dict = token_response
    
        # Convert the internal response to Anthropic API format
        return {"input_tokens": _token_response_dict.get("total_tokens", 0)}
        
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(
            "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(str(e))
        )
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )
