"""
Transformation module for Anthropic ↔ OpenAI format conversion.

This module provides bidirectional transformation between Anthropic Messages API
format and OpenAI Chat Completions API format, enabling seamless routing of
Anthropic requests to any OpenAI-compatible provider.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AnthropicToOpenAITransformer:
    """
    Transform Anthropic Messages API requests to OpenAI Chat Completions format.
    
    This transformer handles:
    - Message format conversion (content blocks → string/tool_calls)
    - Tool definition conversion (input_schema → parameters)
    - Tool choice conversion (Anthropic → OpenAI format)
    - System message handling
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def transform_messages(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[Union[str, List[Dict[str, Any]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform Anthropic messages to OpenAI format.
        
        Handles:
        - Text content blocks → string content
        - tool_use blocks → tool_calls
        - tool_result blocks → tool messages
        - image blocks → image_url format
        - system parameter → system message
        
        Args:
            messages: List of Anthropic messages
            system: Optional system message (string or list of text blocks)
            
        Returns:
            List of OpenAI-formatted messages
        """
        openai_messages = []
        
        # Track valid tool call IDs from assistant messages
        valid_tool_call_ids = set()
        
        # Handle system messages first
        if system:
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                # Extract text from text blocks
                text_parts = [
                    item.get("text", "")
                    for item in system
                    if item.get("type") == "text"
                ]
                if text_parts:
                    openai_messages.append(
                        {"role": "system", "content": "\n".join(text_parts)}
                    )
        
        # First pass: collect all valid tool call IDs from assistant messages
        for message in messages:
            if message["role"] == "assistant" and isinstance(message["content"], list):
                tool_call_parts = [
                    c for c in message["content"] if c.get("type") == "tool_use"
                ]
                for tool in tool_call_parts:
                    tool_id = tool.get("id")
                    if tool_id:
                        valid_tool_call_ids.add(tool_id)
        
        # Process conversation messages with proper tool call handling
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                self._transform_user_message(
                    content, openai_messages, valid_tool_call_ids
                )
            elif role == "assistant":
                self._transform_assistant_message(
                    content, openai_messages, valid_tool_call_ids
                )
            else:
                # Handle other roles
                if isinstance(content, str):
                    openai_messages.append({"role": role, "content": content})
                else:
                    openai_messages.append({"role": role, "content": str(content)})
        
        return openai_messages
    
    def _transform_user_message(
        self,
        content: Union[str, List[Dict[str, Any]]],
        openai_messages: List[Dict[str, Any]],
        valid_tool_call_ids: set
    ) -> None:
        """Transform user message content."""
        if isinstance(content, str):
            openai_messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            # Handle user messages with text and tool results
            text_parts = [c for c in content if c.get("type") == "text"]
            tool_result_parts = [
                c for c in content if c.get("type") == "tool_result"
            ]
            image_parts = [c for c in content if c.get("type") == "image"]
            
            # Process tool results first - each needs to be a separate tool message
            for tool_result in tool_result_parts:
                tool_use_id = tool_result.get("tool_use_id", "")
                if tool_use_id:
                    # Validate that this tool result references a valid tool call
                    if tool_use_id not in valid_tool_call_ids:
                        self.logger.warning(
                            f"Tool result references invalid tool_call_id: {tool_use_id}. "
                            f"Valid IDs: {valid_tool_call_ids}"
                        )
                        # Skip invalid tool results to prevent OpenAI API errors
                        continue
                    
                    tool_content = self._extract_tool_result_content(tool_result)
                    openai_messages.append(
                        {
                            "role": "tool",
                            "content": tool_content,
                            "tool_call_id": tool_use_id,
                        }
                    )
                else:
                    self.logger.warning("Tool result missing tool_use_id, skipping")
            
            # Handle text and image parts as user message
            if text_parts or image_parts:
                user_content = self._build_user_content(text_parts, image_parts)
                if user_content:
                    openai_messages.append({"role": "user", "content": user_content})
    
    def _extract_tool_result_content(self, tool_result: Dict[str, Any]) -> str:
        """Extract content from tool_result block."""
        tool_content = tool_result.get("content", "")
        
        if isinstance(tool_content, dict):
            # Handle dict content with proper text extraction
            if tool_content.get("type") == "text":
                return tool_content.get("text", "")
            else:
                try:
                    return json.dumps(tool_content)
                except Exception:
                    return str(tool_content)
        elif isinstance(tool_content, list):
            # Extract text from content blocks
            extracted_content = ""
            for content_block in tool_content:
                if isinstance(content_block, dict):
                    if content_block.get("type") == "text":
                        extracted_content += content_block.get("text", "") + "\n"
                    else:
                        try:
                            extracted_content += json.dumps(content_block) + "\n"
                        except Exception:
                            extracted_content += str(content_block) + "\n"
                else:
                    extracted_content += str(content_block) + "\n"
            return extracted_content.strip()
        elif not isinstance(tool_content, str):
            return str(tool_content)
        
        return tool_content
    
    def _build_user_content(
        self,
        text_parts: List[Dict[str, Any]],
        image_parts: List[Dict[str, Any]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Build user content from text and image parts."""
        # If only text, return as string
        if text_parts and not image_parts:
            text_content = "\n".join([t.get("text", "") for t in text_parts])
            return text_content if text_content.strip() else None
        
        # If images, build content array
        if image_parts:
            content_array = []
            
            # Add text parts
            for text_part in text_parts:
                text = text_part.get("text", "")
                if text.strip():
                    content_array.append({"type": "text", "text": text})
            
            # Add image parts
            for image_part in image_parts:
                image_content = self._transform_image_block(image_part)
                if image_content:
                    content_array.append(image_content)
            
            return content_array if content_array else None
        
        return None
    
    def _transform_image_block(self, image_block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform Anthropic image block to OpenAI format."""
        source = image_block.get("source", {})
        source_type = source.get("type")
        
        if source_type == "base64":
            media_type = source.get("media_type", "image/jpeg")
            data = source.get("data", "")
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{data}"
                }
            }
        elif source_type == "url":
            url = source.get("url", "")
            return {
                "type": "image_url",
                "image_url": {"url": url}
            }
        
        return None
    
    def _transform_assistant_message(
        self,
        content: Union[str, List[Dict[str, Any]]],
        openai_messages: List[Dict[str, Any]],
        valid_tool_call_ids: set
    ) -> None:
        """Transform assistant message content."""
        if isinstance(content, str):
            openai_messages.append({"role": "assistant", "content": content})
        elif isinstance(content, list):
            # Handle assistant messages with text and tool calls
            text_parts = [c for c in content if c.get("type") == "text"]
            tool_call_parts = [
                c for c in content if c.get("type") == "tool_use"
            ]
            
            # Combine text parts
            text_content = "\n".join([t.get("text", "") for t in text_parts])
            
            if tool_call_parts:
                # Create tool_calls for OpenAI format
                tool_calls = []
                for tool in tool_call_parts:
                    tool_id = tool.get("id", f"call_{int(time.time() * 1000)}")
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool.get("name", ""),
                                "arguments": json.dumps(tool.get("input", {})),
                            },
                        }
                    )
                    # Add to valid tool call IDs
                    valid_tool_call_ids.add(tool_id)
                
                # Add assistant message with tool calls
                openai_messages.append(
                    {
                        "role": "assistant",
                        "content": text_content if text_content.strip() else None,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                # Just text content
                if text_content.strip():
                    openai_messages.append(
                        {"role": "assistant", "content": text_content}
                    )
    
    def transform_tools(
        self,
        tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform Anthropic tools to OpenAI format.
        
        Converts:
        - input_schema → parameters
        - Anthropic tool format → OpenAI function format
        
        Args:
            tools: List of Anthropic tool definitions
            
        Returns:
            List of OpenAI-formatted tools
        """
        openai_tools = []
        
        for tool in tools:
            if "function" in tool:
                # Already in OpenAI format
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"].get("parameters", {}),
                        },
                    }
                )
            else:
                # Convert from Anthropic format
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        },
                    }
                )
        
        return openai_tools
    
    def transform_tool_choice(
        self,
        tool_choice: Union[str, Dict[str, Any]]
    ) -> Union[str, Dict[str, Any]]:
        """
        Transform Anthropic tool_choice to OpenAI format.
        
        Mappings:
        - "auto" → "auto"
        - "any" → "required"
        - {"type": "tool", "name": "x"} → {"type": "function", "function": {"name": "x"}}
        
        Args:
            tool_choice: Anthropic tool_choice specification
            
        Returns:
            OpenAI-formatted tool_choice
        """
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "tool":
                # Convert Anthropic tool choice to OpenAI format
                return {
                    "type": "function",
                    "function": {"name": tool_choice.get("name")},
                }
            elif tool_choice.get("type") == "auto":
                return "auto"
            elif tool_choice.get("type") == "any":
                return "required"
        elif isinstance(tool_choice, str):
            if tool_choice == "auto":
                return "auto"
            elif tool_choice == "any":
                return "required"
        
        return tool_choice


class OpenAIToAnthropicTransformer:
    """
    Transform OpenAI Chat Completions responses to Anthropic Messages API format.
    
    This transformer handles:
    - Response format conversion
    - Content block generation
    - Stop reason mapping
    - Usage data transformation
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def transform_response(
        self,
        openai_response: Dict[str, Any],
        original_model: str
    ) -> Dict[str, Any]:
        """
        Transform OpenAI response to Anthropic format.
        
        Converts:
        - text content → text content block
        - tool_calls → tool_use content blocks
        - finish_reason → stop_reason
        - usage → Anthropic usage format
        
        Args:
            openai_response: OpenAI response dictionary
            original_model: Original Anthropic model name
            
        Returns:
            Anthropic-formatted response dictionary
        """
        timestamp = int(time.time() * 1000)
        
        # Extract usage data
        usage_data = openai_response.get("usage", {}) or {}
        usage = {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0),
        }
        
        response_id = openai_response.get("id", f"msg_{timestamp}")
        
        # Handle choices
        choices = openai_response.get("choices", [])
        if not choices:
            # If no choices, create a default choice with empty message
            choice = {
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
            choices = [choice]
        
        # Process all choices and combine content blocks
        content_blocks = []
        combined_stop_reason = None
        assistant_role = "assistant"
        
        for choice in choices:
            message = choice.get("message", {})
            choice_stop_reason = self.map_stop_reason(
                choice.get("finish_reason", "")
            ) or "end_turn"
            
            # Use the first non-null stop reason we encounter
            if combined_stop_reason is None:
                combined_stop_reason = choice_stop_reason
            
            # Update role if available
            if message.get("role"):
                assistant_role = message["role"]
            
            # Handle text content
            text_content = message.get("content")
            if text_content and text_content.strip():
                content_blocks.append({"type": "text", "text": text_content})
            
            # Handle tool calls
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    tool_block = self._convert_tool_call_to_content_block(
                        tool_call, timestamp
                    )
                    content_blocks.append(tool_block)
        
        # Ensure at least one content block
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        
        # Use combined stop reason or default
        final_stop_reason = combined_stop_reason or "end_turn"
        
        return {
            "id": response_id,
            "type": "message",
            "role": assistant_role,
            "content": content_blocks,
            "model": original_model,
            "stop_reason": final_stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }
    
    def _convert_tool_call_to_content_block(
        self,
        tool_call: Dict[str, Any],
        timestamp: int
    ) -> Dict[str, Any]:
        """Convert OpenAI tool call to Anthropic tool_use content block."""
        if not isinstance(tool_call, dict):
            self.logger.error(f"Tool call must be a dictionary, got {type(tool_call)}")
            raise ValueError(f"Tool call must be a dictionary, got {type(tool_call)}")
        
        function_data = tool_call.get("function", {})
        if not isinstance(function_data, dict):
            self.logger.warning(f"Tool call function data invalid: {function_data}")
            function_data = {}
        
        # Extract tool information
        tool_name = function_data.get("name", "")
        tool_id = tool_call.get("id", f"toolu_{timestamp}")
        arguments = function_data.get("arguments", "{}")
        
        # Parse arguments
        try:
            if isinstance(arguments, str):
                if arguments.strip():
                    parsed_input = json.loads(arguments)
                else:
                    parsed_input = {}
            elif isinstance(arguments, dict):
                parsed_input = arguments
            elif arguments is None:
                parsed_input = {}
            else:
                self.logger.error(
                    f"Unexpected arguments type: {type(arguments)}, "
                    f"cannot convert to valid tool input"
                )
                parsed_input = {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in tool arguments: {e}")
            self.logger.error(f"Raw arguments: {repr(arguments)}")
            parsed_input = {}
        except Exception as e:
            self.logger.error(f"Unexpected error parsing tool arguments: {e}")
            self.logger.error(f"Raw arguments: {repr(arguments)}")
            parsed_input = {}
        
        return {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": parsed_input,
        }
    
    def map_stop_reason(self, openai_finish_reason: str) -> Optional[str]:
        """
        Map OpenAI finish_reason to Anthropic stop_reason.
        
        Mappings:
        - "stop" → "end_turn"
        - "length" → "max_tokens"
        - "tool_calls" → "tool_use"
        - "content_filter" → "stop_sequence"
        
        Args:
            openai_finish_reason: OpenAI finish_reason value
            
        Returns:
            Anthropic stop_reason value or None
        """
        if not openai_finish_reason:
            return None
        
        stop_reason_mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
            "function_call": "tool_use",
            # Also handle cases where Anthropic format is already used
            "end_turn": "end_turn",
            "max_tokens": "max_tokens",
            "tool_use": "tool_use",
            "stop_sequence": "stop_sequence",
        }
        
        return stop_reason_mapping.get(openai_finish_reason, "end_turn")
