"""
Enhanced Anthropic API endpoints with full compatibility support.
Provides comprehensive Anthropic API compatibility including model mapping,
enhanced validation, streaming, and additional endpoints.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse

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
from litellm.types.utils import TokenCountResponse

# Import our enhanced modules
from .validation import (
    validate_messages,
    validate_tools,
    validate_system_message,
    InvalidRequestError,
)
from .model_mapping import anthropic_model_mapper
from .config import initialize_anthropic_endpoints

router = APIRouter()


@router.post(
    "/v1/messages", 
    tags=["Anthropic API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(  # noqa: PLR0915
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enhanced Anthropic Messages API endpoint with full compatibility support.
    
    Supports:
    - Comprehensive message validation
    - Model tier mapping and resolution
    - Enhanced streaming with proper SSE format
    - Tool calling with validation
    - Multiple provider backends
    
    This endpoint provides full compatibility with Claude Code and other Anthropic clients.
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
    
    # Enhanced validation
    try:
        # Validate required fields
        if not data.get("model"):
            raise InvalidRequestError("model is required")
        if not data.get("messages"):
            raise InvalidRequestError("messages is required")
            
        # Validate message structure
        validate_messages(data["messages"])
        
        # Validate tools if provided
        if data.get("tools"):
            validate_tools(data["tools"])
            
        # Validate system message if provided
        if data.get("system"):
            validate_system_message(data["system"])
            
        # Apply model mapping if configured
        original_model = data["model"]
        verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: Original model request: {original_model}")
        
        try:
            # Log the current state of the model mapper
            verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: anthropic_model_mapper available providers: {list(anthropic_model_mapper.provider_configs.keys())}")
            verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: default provider: {anthropic_model_mapper.default_provider}")
            
            # Get the tier for debugging
            tier = anthropic_model_mapper.get_anthropic_model_tier(original_model)
            verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: {original_model} -> tier: {tier}")
            
            # Attempt model resolution
            resolved_model = anthropic_model_mapper.resolve_model(original_model)
            data["model"] = resolved_model
            verbose_proxy_logger.info(f"✅ MODEL MAPPING SUCCESS: {original_model} -> {resolved_model}")
        except ValueError as e:
            verbose_proxy_logger.warning(f"❌ MODEL MAPPING FAILED for {original_model}: {e}")
            verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: Keeping original model: {original_model}")
            # Keep original model if mapping fails
        except Exception as e:
            verbose_proxy_logger.error(f"❌ MODEL MAPPING ERROR for {original_model}: {e}")
            verbose_proxy_logger.info(f"🔍 MODEL MAPPING DEBUG: Keeping original model due to error: {original_model}")
            import traceback
            verbose_proxy_logger.debug(f"MODEL MAPPING ERROR TRACEBACK: {traceback.format_exc()}")
            
        # Model mapping completed - proceeding with request processing
            
    except InvalidRequestError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e)
                }
            }
        )
    
    try:
        data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        data = await add_litellm_data_to_request(
            data=data,  # type: ignore
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            data["temperature"] = user_temperature
        if user_request_timeout:
            data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            data["max_tokens"] = user_max_tokens
        if user_api_base:
            data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data before calling the model
        data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict, data=data, call_type="text_completion"
        )

        tasks = []
        tasks.append(
            proxy_logging_obj.during_call_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                call_type=ProxyBaseLLMRequestProcessing._get_pre_call_type(
                    route_type="anthropic_messages"  # type: ignore
                ),
            )
        )

        ### ROUTE THE REQUESTs ###
        router_model_names = llm_router.model_names if llm_router is not None else []

        # skip router if user passed their key
        if (
            llm_router is not None and data["model"] in router_model_names
        ):  # model in router model list
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None
            and llm_router.model_group_alias is not None
            and data["model"] in llm_router.model_group_alias
        ):  # model set in model_group_alias
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None and data["model"] in llm_router.deployment_names
        ):  # model in router deployments, calling a specific deployment on the router
            llm_coro = llm_router.aanthropic_messages(**data, specific_deployment=True)
        elif (
            llm_router is not None and llm_router.has_model_id(data["model"])
        ):  # model in router model list
            llm_coro = llm_router.aanthropic_messages(**data)
        elif (
            llm_router is not None
            and data["model"] not in router_model_names
            and (
                llm_router.default_deployment is not None
                or len(llm_router.pattern_router.patterns) > 0
            )
        ):  # model in router deployments, calling a specific deployment on the router
            llm_coro = llm_router.aanthropic_messages(**data)
        elif user_model is not None:  # `litellm --model <your-model-name>`
            llm_coro = litellm.anthropic_messages(**data)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "completion: Invalid model name passed in model="
                    + data.get("model", "")
                },
            )

        tasks.append(llm_coro)

        # wait for call to end
        llm_responses = asyncio.gather(
            *tasks
        )  # run the moderation check in parallel to the actual llm api call

        responses = await llm_responses

        response = responses[1]

        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data.get("litellm_call_id", ""), status="success"
            )
        )

        verbose_proxy_logger.debug("final response: %s", response)

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        if (
            "stream" in data and data["stream"] is True
        ):  # use generate_responses to stream responses
            selected_data_generator = (
                ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                    request_data=data,
                    proxy_logging_obj=proxy_logging_obj,
                )
            )

            return await create_streaming_response(
                generator=selected_data_generator,
                media_type="text/event-stream",
                headers=dict(fastapi_response.headers),
            )

        ### CALL HOOKS ### - modify outgoing data
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response # type: ignore
        )

        verbose_proxy_logger.debug("\nResponse from Litellm:\n{}".format(response))
        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.exception(
            "litellm.proxy.proxy_server.anthropic_response(): Exception occured - {}".format(
                str(e)
            )
        )
        error_msg = f"{str(e)}"
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            code=getattr(e, "status_code", 500),
        )


@router.get(
    "/v1/models",
    tags=["Anthropic API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_models(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List available Anthropic models.
    
    Returns a list of available models in Anthropic API format,
    including both standard Anthropic models and any custom mappings.
    """
    try:
        models = anthropic_model_mapper.list_available_models()
        return {"data": models, "object": "list"}
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )


@router.post(
    "/v1/count_tokens",
    tags=["Anthropic API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def count_tokens_enhanced(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enhanced token counting for Anthropic Messages API format.
    
    This endpoint accepts the same parameters as the /v1/messages endpoint
    but returns token counts instead of generating a response.
    
    Supports:
    - Message validation
    - Model mapping
    - Tool definitions
    - System messages
    
    Example usage:
    ```
    curl -X POST "http://localhost:4000/v1/count_tokens" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer your-key" \
      -d '{
        "model": "claude-3-sonnet-20240229",
        "messages": [{"role": "user", "content": "Hello Claude!"}],
        "system": "You are a helpful assistant."
      }'
    ```
    
    Returns: {"input_tokens": <number>}
    """
    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}
        
        # Enhanced validation
        try:
            # Validate required fields
            if not data.get("model"):
                raise InvalidRequestError("model is required")
            if not data.get("messages"):
                raise InvalidRequestError("messages is required")
                
            # Validate message structure
            validate_messages(data["messages"])
            
            # Validate tools if provided
            if data.get("tools"):
                validate_tools(data["tools"])
                
            # Validate system message if provided
            if data.get("system"):
                validate_system_message(data["system"])
                
        except InvalidRequestError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "type": "invalid_request_error", 
                        "message": str(e)
                    }
                }
            )
        
        # Apply model mapping
        original_model = data["model"]
        try:
            resolved_model = anthropic_model_mapper.resolve_model(original_model)
            data["model"] = resolved_model
        except ValueError:
            # Keep original model if mapping fails
            pass
        
        # Use LiteLLM's token counting
        try:
            token_count = litellm.token_counter(
                model=data["model"],
                messages=data["messages"],
                system=data.get("system"),
                tools=data.get("tools"),
            )
            return {"input_tokens": token_count}
        except Exception as e:
            verbose_proxy_logger.debug(f"Token counting failed with LiteLLM, falling back: {e}")
            # Fallback to approximate counting
            total_chars = 0
            for message in data["messages"]:
                if isinstance(message.get("content"), str):
                    total_chars += len(message["content"])
                elif isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total_chars += len(block.get("text", ""))
            
            if data.get("system"):
                if isinstance(data["system"], str):
                    total_chars += len(data["system"])
                elif isinstance(data["system"], list):
                    for block in data["system"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total_chars += len(block.get("text", ""))
            
            # Rough approximation: ~4 chars per token
            estimated_tokens = total_chars // 4
            return {"input_tokens": estimated_tokens}
            
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error in count_tokens: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )


@router.post(
    "/v1/count_request_tokens", 
    tags=["Anthropic API"],
    dependencies=[Depends(user_api_key_auth)],
)
async def count_request_tokens(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Count tokens for a complete request including all parameters.
    
    Similar to count_tokens but provides more detailed token breakdown
    when available.
    """
    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}
        
        # Reuse the enhanced count_tokens logic
        result = await count_tokens_enhanced(request, user_api_key_dict)
        
        # Add additional metadata if available
        result["request_tokens"] = result["input_tokens"]
        return result
        
    except Exception as e:
        verbose_proxy_logger.exception(f"Error in count_request_tokens: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )


@router.post(
    "/v1/check_context_limit",
    tags=["Anthropic API"], 
    dependencies=[Depends(user_api_key_auth)],
)
async def check_context_limit(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Check if a request fits within the model's context limit.
    
    Returns information about whether the request would exceed
    the model's maximum context length.
    
    Returns:
    {
        "fits_context": true/false,
        "input_tokens": <number>,
        "max_context_tokens": <number>,
        "remaining_tokens": <number>
    }
    """
    try:
        request_data = await _read_request_body(request=request)
        data: dict = {**request_data}
        
        # Get token count using our enhanced counter
        token_result = await count_tokens_enhanced(request, user_api_key_dict)
        input_tokens = token_result["input_tokens"]
        
        # Get model context limit
        model_name = data.get("model", "")
        max_tokens = data.get("max_tokens", 1024)
        
        # Default context limits for common models
        context_limits = {
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-haiku-20241022": 200000,
        }
        
        # Try to resolve the original Anthropic model name for context limit lookup
        anthropic_model = model_name
        for anthro_model, tier in anthropic_model_mapper.anthropic_models.items():
            try:
                if anthropic_model_mapper.resolve_model(anthro_model) == model_name:
                    anthropic_model = anthro_model
                    break
            except ValueError:
                continue
        
        max_context_tokens = context_limits.get(anthropic_model, 100000)  # Default fallback
        
        # Calculate if request fits
        total_required_tokens = input_tokens + max_tokens
        fits_context = total_required_tokens <= max_context_tokens
        remaining_tokens = max_context_tokens - input_tokens
        
        return {
            "fits_context": fits_context,
            "input_tokens": input_tokens,
            "max_context_tokens": max_context_tokens,
            "remaining_tokens": max(0, remaining_tokens),
            "max_output_tokens": max_tokens,
        }
        
    except Exception as e:
        verbose_proxy_logger.exception(f"Error in check_context_limit: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": f"Internal server error: {str(e)}"}
        )


@router.post(
    "/v1/messages/count_tokens",
    tags=["Anthropic API - Legacy"],
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
