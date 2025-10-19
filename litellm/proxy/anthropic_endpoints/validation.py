"""
Enhanced validation for Anthropic API endpoints.
Provides comprehensive validation for messages, tools, and content blocks.
"""

import json
from typing import Any, Dict, List, Set

from litellm.proxy._types import ProxyException


class InvalidRequestError(ProxyException):
    """Custom exception for invalid Anthropic API requests."""
    
    def __init__(self, message: str, code: str = "invalid_request_error", status_code: int = 400):
        super().__init__(
            message=message,
            type=code,
            param=None,
            code=status_code
        )


def validate_tools(tools: List[Dict[str, Any]]) -> None:
    """Validate tool definitions to ensure they have required fields and correct structure."""
    if not tools:
        return

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise InvalidRequestError(
                f"Tool at index {i} must be a dictionary, got {type(tool)}"
            )

        # Handle OpenAI format
        if "function" in tool:
            function = tool["function"]
            if not isinstance(function, dict):
                raise InvalidRequestError(
                    f"Tool at index {i} function must be a dictionary, got {type(function)}"
                )

            if "name" not in function:
                raise InvalidRequestError(
                    f"Tool at index {i} function must have a 'name' field"
                )

            if "description" not in function:
                raise InvalidRequestError(
                    f"Tool '{function['name']}' at index {i} must have a 'description' field"
                )

            # Validate parameters schema if present
            if "parameters" in function:
                validate_parameters_schema(function["parameters"], function["name"])

        # Handle Anthropic format
        else:
            if "name" not in tool:
                raise InvalidRequestError(f"Tool at index {i} must have a 'name' field")

            if "description" not in tool:
                raise InvalidRequestError(
                    f"Tool '{tool['name']}' at index {i} must have a 'description' field"
                )

            # Validate input_schema if present
            if "input_schema" in tool:
                validate_parameters_schema(tool["input_schema"], tool["name"])


def validate_parameters_schema(schema: Dict[str, Any], tool_name: str) -> None:
    """Validate the parameters/input_schema of a tool."""
    if not schema:
        return

    if not isinstance(schema, dict):
        raise InvalidRequestError(
            f"Tool '{tool_name}' parameters must be a dictionary, got {type(schema)}"
        )

    if "type" not in schema:
        raise InvalidRequestError(
            f"Tool '{tool_name}' parameters must have a 'type' field"
        )

    if schema["type"] != "object":
        raise InvalidRequestError(
            f"Tool '{tool_name}' parameters type must be 'object', got '{schema['type']}'"
        )

    # Validate properties if present
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise InvalidRequestError(
                f"Tool '{tool_name}' properties must be a dictionary, got {type(schema['properties'])}"
            )

        # Validate each property
        for prop_name, prop_def in schema["properties"].items():
            if not isinstance(prop_def, dict):
                raise InvalidRequestError(
                    f"Tool '{tool_name}' property '{prop_name}' must be a dictionary, got {type(prop_def)}"
                )

    # Validate required fields if present
    if "required" in schema:
        if not isinstance(schema["required"], list):
            raise InvalidRequestError(
                f"Tool '{tool_name}' required must be a list, got {type(schema['required'])}"
            )

        if "properties" in schema:
            properties = schema["properties"]
            for required_field in schema["required"]:
                if required_field not in properties:
                    raise InvalidRequestError(
                        f"Tool '{tool_name}' required field '{required_field}' is not defined in properties"
                    )


def validate_messages(messages: List[Dict[str, Any]]) -> None:
    """
    Validate message structure and content blocks to ensure they conform to Anthropic API spec.

    Args:
        messages: List of message dictionaries to validate

    Raises:
        InvalidRequestError: If validation fails
    """
    if not isinstance(messages, list):
        raise InvalidRequestError("messages must be a list")

    if not messages:
        raise InvalidRequestError("messages cannot be empty")

    valid_tool_use_ids = set()

    # First pass: collect all valid tool_use IDs from assistant messages
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidRequestError(f"Message at index {i} must be a dictionary")

        role = message.get("role")
        if not role:
            raise InvalidRequestError(f"Message at index {i} must have a 'role' field")

        if role not in ["user", "assistant", "system"]:
            raise InvalidRequestError(
                f"Message at index {i} has invalid role '{role}'. Must be 'user', 'assistant', or 'system'"
            )

        content = message.get("content")
        if content is None:
            raise InvalidRequestError(
                f"Message at index {i} must have a 'content' field"
            )

        # Validate content structure and collect tool_use IDs
        if role == "assistant":
            valid_tool_use_ids.update(
                validate_content_blocks(content, f"message {i}", role)
            )
        else:
            validate_content_blocks(content, f"message {i}", role)

    # Second pass: validate tool_result references
    for i, message in enumerate(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                for j, content_block in enumerate(content):
                    if (
                        isinstance(content_block, dict)
                        and content_block.get("type") == "tool_result"
                    ):
                        tool_use_id = content_block.get("tool_use_id")
                        if not tool_use_id:
                            raise InvalidRequestError(
                                f"tool_result at message {i}, content block {j} must have 'tool_use_id'"
                            )

                        if tool_use_id not in valid_tool_use_ids:
                            raise InvalidRequestError(
                                f"tool_result at message {i}, content block {j} references invalid tool_use_id '{tool_use_id}'. Valid IDs: {list(valid_tool_use_ids)}"
                            )


def validate_content_blocks(content: Any, context: str, role: str) -> Set[str]:
    """
    Validate content blocks structure and return any tool_use IDs found.

    Args:
        content: Content to validate (string or list of content blocks)
        context: Context string for error messages
        role: Message role for context-specific validation

    Returns:
        Set of tool_use IDs found in the content

    Raises:
        InvalidRequestError: If validation fails
    """
    tool_use_ids = set()

    if isinstance(content, str):
        # Simple text content is always valid
        if not content.strip() and role != "assistant":
            raise InvalidRequestError(f"{context} cannot have empty content")
        return tool_use_ids

    elif isinstance(content, list):
        if not content:
            raise InvalidRequestError(f"{context} content list cannot be empty")

        for i, block in enumerate(content):
            if not isinstance(block, dict):
                raise InvalidRequestError(
                    f"{context} content block {i} must be a dictionary"
                )

            block_type = block.get("type")
            if not block_type:
                raise InvalidRequestError(
                    f"{context} content block {i} must have a 'type' field"
                )

            # Validate based on content block type
            if block_type == "text":
                validate_text_block(block, f"{context} content block {i}")

            elif block_type == "tool_use":
                if role != "assistant":
                    raise InvalidRequestError(
                        f"{context} content block {i}: tool_use blocks are only allowed in assistant messages"
                    )
                tool_id = validate_tool_use_block(block, f"{context} content block {i}")
                if tool_id:
                    tool_use_ids.add(tool_id)

            elif block_type == "tool_result":
                if role != "user":
                    raise InvalidRequestError(
                        f"{context} content block {i}: tool_result blocks are only allowed in user messages"
                    )
                validate_tool_result_block(block, f"{context} content block {i}")

            elif block_type == "image":
                if role not in ["user", "assistant"]:
                    raise InvalidRequestError(
                        f"{context} content block {i}: image blocks are only allowed in user or assistant messages"
                    )
                validate_image_block(block, f"{context} content block {i}")

            else:
                raise InvalidRequestError(
                    f"{context} content block {i} has unsupported type '{block_type}'. Supported types: text, tool_use, tool_result, image"
                )

    else:
        raise InvalidRequestError(
            f"{context} content must be a string or list of content blocks"
        )

    return tool_use_ids


def validate_text_block(block: Dict[str, Any], context: str) -> None:
    """Validate a text content block."""
    text = block.get("text")
    if text is None:
        raise InvalidRequestError(f"{context}: text blocks must have a 'text' field")

    if not isinstance(text, str):
        raise InvalidRequestError(
            f"{context}: text field must be a string, got {type(text)}"
        )


def validate_tool_use_block(block: Dict[str, Any], context: str) -> str:
    """Validate a tool_use content block and return the tool ID."""
    tool_id = block.get("id")
    if not tool_id:
        raise InvalidRequestError(f"{context}: tool_use blocks must have an 'id' field")

    if not isinstance(tool_id, str):
        raise InvalidRequestError(
            f"{context}: tool_use id must be a string, got {type(tool_id)}"
        )

    name = block.get("name")
    if not name:
        raise InvalidRequestError(
            f"{context}: tool_use blocks must have a 'name' field"
        )

    if not isinstance(name, str):
        raise InvalidRequestError(
            f"{context}: tool_use name must be a string, got {type(name)}"
        )

    # input field is optional but if present should be a dict
    tool_input = block.get("input")
    if tool_input is not None and not isinstance(tool_input, dict):
        raise InvalidRequestError(
            f"{context}: tool_use input must be a dictionary if provided, got {type(tool_input)}"
        )

    return tool_id


def validate_tool_result_block(block: Dict[str, Any], context: str) -> None:
    """Validate a tool_result content block."""
    tool_use_id = block.get("tool_use_id")
    if not tool_use_id:
        raise InvalidRequestError(
            f"{context}: tool_result blocks must have a 'tool_use_id' field"
        )

    if not isinstance(tool_use_id, str):
        raise InvalidRequestError(
            f"{context}: tool_use_id must be a string, got {type(tool_use_id)}"
        )

    # is_error field is optional but if present should be a boolean
    is_error = block.get("is_error")
    if is_error is not None and not isinstance(is_error, bool):
        raise InvalidRequestError(
            f"{context}: tool_result is_error must be a boolean if provided, got {type(is_error)}"
        )


def validate_image_block(block: Dict[str, Any], context: str) -> None:
    """Validate an image content block."""
    source = block.get("source")
    if not source:
        raise InvalidRequestError(f"{context}: image blocks must have a 'source' field")

    if not isinstance(source, dict):
        raise InvalidRequestError(
            f"{context}: image source must be a dictionary, got {type(source)}"
        )

    source_type = source.get("type")
    if source_type != "base64":
        raise InvalidRequestError(
            f"{context}: image source type must be 'base64', got '{source_type}'"
        )

    media_type = source.get("media_type")
    if not media_type:
        raise InvalidRequestError(
            f"{context}: image source must have a 'media_type' field"
        )

    if not media_type.startswith("image/"):
        raise InvalidRequestError(
            f"{context}: image media_type must start with 'image/', got '{media_type}'"
        )

    data = source.get("data")
    if not data:
        raise InvalidRequestError(f"{context}: image source must have a 'data' field")

    if not isinstance(data, str):
        raise InvalidRequestError(
            f"{context}: image data must be a base64 string, got {type(data)}"
        )


def validate_system_message(system: Any) -> None:
    """
    Validate system message format.

    Args:
        system: System message (string or list of content blocks)

    Raises:
        InvalidRequestError: If validation fails
    """
    if isinstance(system, str):
        if not system.strip():
            raise InvalidRequestError("system message cannot be empty")
    elif isinstance(system, list):
        if not system:
            raise InvalidRequestError("system message list cannot be empty")

        for i, block in enumerate(system):
            if not isinstance(block, dict):
                raise InvalidRequestError(
                    f"system message block {i} must be a dictionary"
                )

            block_type = block.get("type")
            if not block_type:
                raise InvalidRequestError(
                    f"system message block {i} must have a 'type' field"
                )

            if block_type == "text":
                validate_text_block(block, f"system message block {i}")
            else:
                raise InvalidRequestError(
                    f"system message block {i} has unsupported type '{block_type}'. Only 'text' blocks are allowed in system messages"
                )
    else:
        raise InvalidRequestError(
            f"system message must be a string or list of content blocks, got {type(system)}"
        )