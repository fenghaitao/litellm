"""
Validation module for Anthropic Messages API requests.

This module provides comprehensive validation for Anthropic-formatted requests
to ensure they meet the API specification before transformation and routing.
"""

from typing import Any, Dict, List, Optional, Set, Union


class InvalidRequestError(Exception):
    """
    Anthropic-compatible validation error.
    
    This exception is raised when request validation fails and provides
    error responses in Anthropic's error format.
    """
    
    def __init__(self, message: str, error_type: str = "invalid_request_error"):
        """
        Initialize the validation error.
        
        Args:
            message: Detailed error message
            error_type: Type of error (default: invalid_request_error)
        """
        self.message = message
        self.error_type = error_type
        self.status_code = 400
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to Anthropic's error response format.
        
        Returns:
            Dictionary in Anthropic error format
        """
        return {
            "type": "error",
            "error": {
                "type": self.error_type,
                "message": self.message
            }
        }


def validate_anthropic_request(data: Dict[str, Any]) -> None:
    """
    Validate Anthropic Messages API request.
    
    Performs comprehensive validation of the request structure including:
    - Required fields (model, messages, max_tokens)
    - Message structure and content blocks
    - Tool definitions and schemas
    - Tool result references
    
    Args:
        data: Request data dictionary
        
    Raises:
        InvalidRequestError: If validation fails with Anthropic-compatible format
    """
    # Validate required fields
    if "model" not in data or not data["model"]:
        raise InvalidRequestError("model is required")
    
    if "messages" not in data:
        raise InvalidRequestError("messages is required")
    
    if not isinstance(data["messages"], list):
        raise InvalidRequestError("messages must be an array")
    
    if len(data["messages"]) == 0:
        raise InvalidRequestError("messages must contain at least one message")
    
    # Validate max_tokens if provided (Anthropic requires it for most models)
    if "max_tokens" in data:
        max_tokens = data["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise InvalidRequestError("max_tokens must be a positive integer")
    
    # Validate messages and collect tool_use_ids
    valid_tool_use_ids = validate_messages(data["messages"])
    
    # Validate tools if provided
    if "tools" in data and data["tools"] is not None:
        validate_tools(data["tools"])
    
    # Validate tool_choice if provided
    if "tool_choice" in data and data["tool_choice"] is not None:
        validate_tool_choice(data["tool_choice"], data.get("tools"))
    
    # Validate system parameter if provided
    if "system" in data and data["system"] is not None:
        validate_system_message(data["system"])
    
    # Validate temperature if provided
    if "temperature" in data:
        temperature = data["temperature"]
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1:
            raise InvalidRequestError("temperature must be a number between 0 and 1")
    
    # Validate top_p if provided
    if "top_p" in data:
        top_p = data["top_p"]
        if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
            raise InvalidRequestError("top_p must be a number between 0 and 1")
    
    # Validate top_k if provided
    if "top_k" in data:
        top_k = data["top_k"]
        if not isinstance(top_k, int) or top_k < 0:
            raise InvalidRequestError("top_k must be a non-negative integer")
    
    # Validate stream if provided
    if "stream" in data:
        if not isinstance(data["stream"], bool):
            raise InvalidRequestError("stream must be a boolean")


def validate_messages(messages: List[Dict[str, Any]]) -> Set[str]:
    """
    Validate message structure and content blocks.
    
    Validates:
    - Message role (user, assistant)
    - Content structure (string or list of content blocks)
    - Content block types and required fields
    - Tool result references
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Set of valid tool_use_id values found in assistant messages
        
    Raises:
        InvalidRequestError: If message structure is invalid
    """
    if not isinstance(messages, list):
        raise InvalidRequestError("messages must be an array")
    
    if len(messages) == 0:
        raise InvalidRequestError("messages must contain at least one message")
    
    valid_tool_use_ids: Set[str] = set()
    tool_result_references: Set[str] = set()
    
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidRequestError(f"Message at index {idx} must be an object")
        
        # Validate role
        if "role" not in message:
            raise InvalidRequestError(f"Message at index {idx} is missing 'role' field")
        
        role = message["role"]
        if role not in ["user", "assistant"]:
            raise InvalidRequestError(
                f"Message at index {idx} has invalid role '{role}'. Must be 'user' or 'assistant'"
            )
        
        # Validate content
        if "content" not in message:
            raise InvalidRequestError(f"Message at index {idx} is missing 'content' field")
        
        content = message["content"]
        
        # Validate content structure and collect tool_use_ids
        if isinstance(content, str):
            if not content.strip():
                raise InvalidRequestError(
                    f"Message at index {idx} has empty content string"
                )
        elif isinstance(content, list):
            message_tool_use_ids, message_tool_result_refs = validate_content_blocks(
                content, f"Message at index {idx}", role
            )
            valid_tool_use_ids.update(message_tool_use_ids)
            tool_result_references.update(message_tool_result_refs)
        else:
            raise InvalidRequestError(
                f"Message at index {idx} content must be a string or array of content blocks"
            )
    
    # Validate that all tool_result references point to valid tool_use blocks
    invalid_refs = tool_result_references - valid_tool_use_ids
    if invalid_refs:
        valid_ids_str = ", ".join(f"'{id}'" for id in sorted(valid_tool_use_ids))
        invalid_ids_str = ", ".join(f"'{id}'" for id in sorted(invalid_refs))
        raise InvalidRequestError(
            f"tool_result references invalid tool_use_id(s): {invalid_ids_str}. "
            f"Valid IDs: [{valid_ids_str}]"
        )
    
    return valid_tool_use_ids


def validate_content_blocks(
    content: List[Dict[str, Any]], 
    context: str, 
    role: str
) -> tuple[Set[str], Set[str]]:
    """
    Validate content blocks structure.
    
    Validates all content block types:
    - text: requires 'text' field
    - image: requires 'source' field with type and data/url
    - tool_use: requires 'id', 'name', and 'input' fields (assistant only)
    - tool_result: requires 'tool_use_id' and 'content' fields (user only)
    
    Args:
        content: List of content block dictionaries
        context: Context string for error messages (e.g., "Message at index 0")
        role: Message role (user or assistant)
        
    Returns:
        Tuple of (tool_use_ids found, tool_result references found)
        
    Raises:
        InvalidRequestError: If content block structure is invalid
    """
    if not isinstance(content, list):
        raise InvalidRequestError(f"{context}: content must be an array")
    
    if len(content) == 0:
        raise InvalidRequestError(f"{context}: content array must not be empty")
    
    tool_use_ids: Set[str] = set()
    tool_result_refs: Set[str] = set()
    
    for block_idx, block in enumerate(content):
        if not isinstance(block, dict):
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: must be an object"
            )
        
        if "type" not in block:
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: missing 'type' field"
            )
        
        block_type = block["type"]
        
        if block_type == "text":
            validate_text_block(block, context, block_idx)
        
        elif block_type == "image":
            validate_image_block(block, context, block_idx)
        
        elif block_type == "tool_use":
            if role != "assistant":
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}: tool_use blocks are only allowed in assistant messages"
                )
            tool_use_id = validate_tool_use_block(block, context, block_idx)
            if tool_use_id in tool_use_ids:
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}: duplicate tool_use_id '{tool_use_id}'"
                )
            tool_use_ids.add(tool_use_id)
        
        elif block_type == "tool_result":
            if role != "user":
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}: tool_result blocks are only allowed in user messages"
                )
            tool_result_ref = validate_tool_result_block(block, context, block_idx)
            tool_result_refs.add(tool_result_ref)
        
        else:
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: invalid type '{block_type}'. "
                f"Must be one of: text, image, tool_use, tool_result"
            )
    
    return tool_use_ids, tool_result_refs


def validate_text_block(block: Dict[str, Any], context: str, block_idx: int) -> None:
    """
    Validate text content block.
    
    Args:
        block: Content block dictionary
        context: Context string for error messages
        block_idx: Block index for error messages
        
    Raises:
        InvalidRequestError: If text block is invalid
    """
    if "text" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: text blocks must have a 'text' field"
        )
    
    if not isinstance(block["text"], str):
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: 'text' field must be a string"
        )


def validate_image_block(block: Dict[str, Any], context: str, block_idx: int) -> None:
    """
    Validate image content block.
    
    Args:
        block: Content block dictionary
        context: Context string for error messages
        block_idx: Block index for error messages
        
    Raises:
        InvalidRequestError: If image block is invalid
    """
    if "source" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: image blocks must have a 'source' field"
        )
    
    source = block["source"]
    if not isinstance(source, dict):
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: 'source' must be an object"
        )
    
    if "type" not in source:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: image source must have a 'type' field"
        )
    
    source_type = source["type"]
    if source_type == "base64":
        if "media_type" not in source:
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: base64 image source must have 'media_type' field"
            )
        if "data" not in source:
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: base64 image source must have 'data' field"
            )
    elif source_type == "url":
        if "url" not in source:
            raise InvalidRequestError(
                f"{context}, content block {block_idx}: url image source must have 'url' field"
            )
    else:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: image source type must be 'base64' or 'url'"
        )


def validate_tool_use_block(block: Dict[str, Any], context: str, block_idx: int) -> str:
    """
    Validate tool_use content block.
    
    Args:
        block: Content block dictionary
        context: Context string for error messages
        block_idx: Block index for error messages
        
    Returns:
        The tool_use_id from the block
        
    Raises:
        InvalidRequestError: If tool_use block is invalid
    """
    if "id" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use blocks must have an 'id' field"
        )
    
    if not isinstance(block["id"], str) or not block["id"].strip():
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use 'id' must be a non-empty string"
        )
    
    if "name" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use blocks must have a 'name' field"
        )
    
    if not isinstance(block["name"], str) or not block["name"].strip():
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use 'name' must be a non-empty string"
        )
    
    if "input" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use blocks must have an 'input' field"
        )
    
    if not isinstance(block["input"], dict):
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_use 'input' must be an object"
        )
    
    return block["id"]


def validate_tool_result_block(block: Dict[str, Any], context: str, block_idx: int) -> str:
    """
    Validate tool_result content block.
    
    Args:
        block: Content block dictionary
        context: Context string for error messages
        block_idx: Block index for error messages
        
    Returns:
        The tool_use_id referenced by this tool_result
        
    Raises:
        InvalidRequestError: If tool_result block is invalid
    """
    if "tool_use_id" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_result blocks must have a 'tool_use_id' field"
        )
    
    if not isinstance(block["tool_use_id"], str) or not block["tool_use_id"].strip():
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_result 'tool_use_id' must be a non-empty string"
        )
    
    if "content" not in block:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_result blocks must have a 'content' field"
        )
    
    # Content can be string or array of content blocks
    content = block["content"]
    if isinstance(content, str):
        pass  # Valid
    elif isinstance(content, list):
        # Validate nested content blocks (text or image only)
        for nested_idx, nested_block in enumerate(content):
            if not isinstance(nested_block, dict):
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}, nested content {nested_idx}: must be an object"
                )
            if "type" not in nested_block:
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}, nested content {nested_idx}: missing 'type' field"
                )
            nested_type = nested_block["type"]
            if nested_type == "text":
                validate_text_block(nested_block, f"{context}, content block {block_idx}", nested_idx)
            elif nested_type == "image":
                validate_image_block(nested_block, f"{context}, content block {block_idx}", nested_idx)
            else:
                raise InvalidRequestError(
                    f"{context}, content block {block_idx}, nested content {nested_idx}: "
                    f"tool_result content can only contain 'text' or 'image' blocks"
                )
    else:
        raise InvalidRequestError(
            f"{context}, content block {block_idx}: tool_result 'content' must be a string or array"
        )
    
    return block["tool_use_id"]


def validate_tools(tools: List[Dict[str, Any]]) -> None:
    """
    Validate tool definitions.
    
    Validates:
    - Tool structure (name, description, input_schema)
    - JSON schema validity for input_schema
    - Required fields in schema
    
    Args:
        tools: List of tool definition dictionaries
        
    Raises:
        InvalidRequestError: If tool definitions are invalid
    """
    if not isinstance(tools, list):
        raise InvalidRequestError("tools must be an array")
    
    if len(tools) == 0:
        raise InvalidRequestError("tools array must not be empty when provided")
    
    tool_names: Set[str] = set()
    
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise InvalidRequestError(f"Tool at index {idx} must be an object")
        
        # Check if already in OpenAI format
        if "function" in tool:
            # OpenAI format - validate function structure
            function = tool["function"]
            if not isinstance(function, dict):
                raise InvalidRequestError(f"Tool at index {idx}: 'function' must be an object")
            
            if "name" not in function:
                raise InvalidRequestError(f"Tool at index {idx}: function must have 'name' field")
            
            if not isinstance(function["name"], str) or not function["name"].strip():
                raise InvalidRequestError(
                    f"Tool at index {idx}: function 'name' must be a non-empty string"
                )
            
            tool_name = function["name"]
        else:
            # Anthropic format - validate Anthropic structure
            if "name" not in tool:
                raise InvalidRequestError(f"Tool at index {idx} must have 'name' field")
            
            if not isinstance(tool["name"], str) or not tool["name"].strip():
                raise InvalidRequestError(
                    f"Tool at index {idx}: 'name' must be a non-empty string"
                )
            
            tool_name = tool["name"]
            
            # Description is optional but should be string if provided
            if "description" in tool and not isinstance(tool["description"], str):
                raise InvalidRequestError(
                    f"Tool at index {idx}: 'description' must be a string"
                )
            
            # Validate input_schema if provided
            if "input_schema" in tool:
                input_schema = tool["input_schema"]
                if not isinstance(input_schema, dict):
                    raise InvalidRequestError(
                        f"Tool at index {idx}: 'input_schema' must be an object"
                    )
                
                # Basic JSON schema validation
                if "type" in input_schema and input_schema["type"] != "object":
                    raise InvalidRequestError(
                        f"Tool at index {idx}: input_schema 'type' must be 'object'"
                    )
        
        # Check for duplicate tool names
        if tool_name in tool_names:
            raise InvalidRequestError(f"Duplicate tool name '{tool_name}'")
        tool_names.add(tool_name)


def validate_tool_choice(
    tool_choice: Union[str, Dict[str, Any]], 
    tools: Optional[List[Dict[str, Any]]]
) -> None:
    """
    Validate tool_choice parameter.
    
    Args:
        tool_choice: Tool choice specification (string or dict)
        tools: List of available tools (for validation)
        
    Raises:
        InvalidRequestError: If tool_choice is invalid
    """
    if isinstance(tool_choice, str):
        if tool_choice not in ["auto", "any", "none"]:
            raise InvalidRequestError(
                f"tool_choice string must be 'auto', 'any', or 'none', got '{tool_choice}'"
            )
    elif isinstance(tool_choice, dict):
        if "type" not in tool_choice:
            raise InvalidRequestError("tool_choice object must have 'type' field")
        
        choice_type = tool_choice["type"]
        
        if choice_type == "tool":
            if "name" not in tool_choice:
                raise InvalidRequestError(
                    "tool_choice with type 'tool' must have 'name' field"
                )
            
            tool_name = tool_choice["name"]
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise InvalidRequestError(
                    "tool_choice 'name' must be a non-empty string"
                )
            
            # Validate that the tool exists in the tools list
            if tools:
                available_tool_names = set()
                for tool in tools:
                    if "function" in tool:
                        available_tool_names.add(tool["function"]["name"])
                    else:
                        available_tool_names.add(tool["name"])
                
                if tool_name not in available_tool_names:
                    raise InvalidRequestError(
                        f"tool_choice references unknown tool '{tool_name}'. "
                        f"Available tools: {sorted(available_tool_names)}"
                    )
        elif choice_type not in ["auto", "any"]:
            raise InvalidRequestError(
                f"tool_choice type must be 'auto', 'any', or 'tool', got '{choice_type}'"
            )
    else:
        raise InvalidRequestError(
            "tool_choice must be a string or object"
        )


def validate_system_message(system: Union[str, List[Dict[str, Any]]]) -> None:
    """
    Validate system parameter.
    
    The system parameter can be:
    - A string
    - An array of text content blocks
    
    Args:
        system: System message (string or list of content blocks)
        
    Raises:
        InvalidRequestError: If system parameter is invalid
    """
    if isinstance(system, str):
        if not system.strip():
            raise InvalidRequestError("system string must not be empty")
    elif isinstance(system, list):
        if len(system) == 0:
            raise InvalidRequestError("system array must not be empty")
        
        for idx, block in enumerate(system):
            if not isinstance(block, dict):
                raise InvalidRequestError(
                    f"system array element {idx} must be an object"
                )
            
            if "type" not in block:
                raise InvalidRequestError(
                    f"system array element {idx} must have 'type' field"
                )
            
            if block["type"] != "text":
                raise InvalidRequestError(
                    f"system array element {idx}: only 'text' type is allowed in system messages"
                )
            
            validate_text_block(block, f"system array element", idx)
    else:
        raise InvalidRequestError(
            "system must be a string or array of text content blocks"
        )
