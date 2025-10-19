"""
Streaming handler for Anthropic SSE format.

This module provides streaming response transformation from OpenAI format
to Anthropic Messages API SSE event format, enabling seamless streaming
for Claude Code and other Anthropic SDK clients.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class AnthropicStreamingHandler:
    """
    Handle streaming response transformation from OpenAI to Anthropic SSE format.
    
    This handler converts OpenAI streaming chunks into Anthropic SSE events,
    maintaining proper state across chunks and ensuring correct event sequencing.
    
    Event sequence:
    1. message_start - Initial message metadata
    2. content_block_start - Start of each content block (text or tool_use)
    3. content_block_delta - Incremental content updates
    4. content_block_stop - End of content block
    5. message_delta - Final message metadata (stop_reason, usage)
    6. message_stop - End of stream
    """
    
    def __init__(self):
        """Initialize the streaming handler with empty state."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.streaming_state = {
            "message_id": None,
            "current_content_block_index": 0,
            "current_block_type": "text",
            "sent_message_start": False,
            "sent_content_block_start": False,
            "tool_use_ids": set(),
            "accumulated_tool_input": "",
            "current_tool_name": None,
            "current_tool_id": None,
            "text_buffer": "",
            "usage_data": {"input_tokens": 0, "output_tokens": 0},
            "stop_reason": None,
        }
    
    async def transform_stream(
        self,
        openai_stream: AsyncIterator,
        model: str,
        system: Optional[Union[str, List[Dict[str, Any]]]] = None,
    ) -> AsyncIterator[bytes]:
        """
        Transform OpenAI stream to Anthropic SSE format.
        
        Yields SSE-formatted events:
        - event: message_start
        - event: content_block_start
        - event: content_block_delta
        - event: content_block_stop
        - event: message_delta
        - event: message_stop
        
        Args:
            openai_stream: AsyncIterator of OpenAI streaming chunks
            model: Model name for the response
            system: Optional system message (for metadata)
            
        Yields:
            bytes: SSE-formatted event data
        """
        try:
            async for chunk in openai_stream:
                # Process the chunk and generate Anthropic events
                events = await self._process_chunk(chunk, model)
                
                # Yield each event as SSE
                for event in events:
                    yield self.format_sse_event(
                        event_type=event["type"],
                        data=event["data"]
                    )
            
            # Send final events
            final_events = self._generate_final_events()
            for event in final_events:
                yield self.format_sse_event(
                    event_type=event["type"],
                    data=event["data"]
                )
                
        except Exception as e:
            self.logger.error(f"Error in stream transformation: {e}")
            # Send error event
            error_event = self._generate_error_event(str(e))
            yield self.format_sse_event(
                event_type=error_event["type"],
                data=error_event["data"]
            )
    
    async def _process_chunk(
        self,
        chunk: Any,
        model: str
    ) -> List[Dict[str, Any]]:
        """
        Process a single OpenAI chunk and generate Anthropic events.
        
        Args:
            chunk: OpenAI streaming chunk (dict or ModelResponseStream)
            model: Model name
            
        Returns:
            List of event dictionaries with 'type' and 'data' keys
        """
        events = []
        
        # Convert chunk to dict if needed
        if hasattr(chunk, "model_dump"):
            chunk_dict = chunk.model_dump()
        elif hasattr(chunk, "dict"):
            chunk_dict = chunk.dict()
        elif isinstance(chunk, dict):
            chunk_dict = chunk
        else:
            self.logger.warning(f"Unexpected chunk type: {type(chunk)}")
            return events
        
        # Extract chunk data
        chunk_id = chunk_dict.get("id")
        choices = chunk_dict.get("choices", [])
        usage = chunk_dict.get("usage")
        
        # Initialize message_id from first chunk
        if not self.streaming_state["message_id"] and chunk_id:
            self.streaming_state["message_id"] = chunk_id
        
        # Send message_start event if not sent yet
        if not self.streaming_state["sent_message_start"]:
            events.append(self._generate_message_start_event(model))
            self.streaming_state["sent_message_start"] = True
        
        # Process choices
        if choices:
            choice = choices[0]  # Use first choice
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")
            
            # Handle content delta (text)
            if "content" in delta and delta["content"]:
                content_events = self._handle_content_delta(delta["content"])
                events.extend(content_events)
            
            # Handle tool calls delta
            if "tool_calls" in delta and delta["tool_calls"]:
                tool_events = self._handle_tool_calls_delta(delta["tool_calls"])
                events.extend(tool_events)
            
            # Handle finish_reason
            if finish_reason:
                self.streaming_state["stop_reason"] = self._map_stop_reason(
                    finish_reason
                )
        
        # Handle usage data
        if usage:
            self.streaming_state["usage_data"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        
        return events
    
    def _handle_content_delta(self, content: str) -> List[Dict[str, Any]]:
        """
        Handle text content delta.
        
        Args:
            content: Text content from delta
            
        Returns:
            List of events for content block
        """
        events = []
        
        # If we were processing a tool call, close it first
        if self.streaming_state["current_block_type"] == "tool_use":
            events.append(self._generate_content_block_stop_event())
            self.streaming_state["current_content_block_index"] += 1
            self.streaming_state["current_block_type"] = "text"
            self.streaming_state["sent_content_block_start"] = False
        
        # Send content_block_start if not sent for current block
        if not self.streaming_state["sent_content_block_start"]:
            events.append(self._generate_content_block_start_event("text"))
            self.streaming_state["sent_content_block_start"] = True
        
        # Send content_block_delta with text
        events.append(self._generate_text_delta_event(content))
        self.streaming_state["text_buffer"] += content
        
        return events
    
    def _handle_tool_calls_delta(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Handle tool calls delta.
        
        Args:
            tool_calls: List of tool call deltas
            
        Returns:
            List of events for tool use blocks
        """
        events = []
        
        for tool_call in tool_calls:
            tool_index = tool_call.get("index", 0)
            tool_id = tool_call.get("id")
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            arguments = function.get("arguments", "")
            
            # Check if this is a new tool call
            is_new_tool = (
                tool_id and tool_id != self.streaming_state["current_tool_id"]
            ) or (
                tool_name and tool_name != self.streaming_state["current_tool_name"]
            )
            
            if is_new_tool:
                # Close previous content block if any
                if self.streaming_state["sent_content_block_start"]:
                    events.append(self._generate_content_block_stop_event())
                    self.streaming_state["current_content_block_index"] += 1
                
                # Initialize new tool call
                if tool_id:
                    self.streaming_state["current_tool_id"] = tool_id
                if tool_name:
                    self.streaming_state["current_tool_name"] = tool_name
                
                self.streaming_state["accumulated_tool_input"] = ""
                self.streaming_state["current_block_type"] = "tool_use"
                
                # Send content_block_start for tool_use
                events.append(
                    self._generate_content_block_start_event(
                        "tool_use",
                        tool_id=self.streaming_state["current_tool_id"],
                        tool_name=self.streaming_state["current_tool_name"]
                    )
                )
                self.streaming_state["sent_content_block_start"] = True
            
            # Accumulate arguments
            if arguments:
                self.streaming_state["accumulated_tool_input"] += arguments
                
                # Send input_json_delta
                events.append(self._generate_input_json_delta_event(arguments))
        
        return events
    
    def _generate_message_start_event(self, model: str) -> Dict[str, Any]:
        """Generate message_start event."""
        message_id = self.streaming_state["message_id"] or f"msg_{int(time.time() * 1000)}"
        
        return {
            "type": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                }
            }
        }
    
    def _generate_content_block_start_event(
        self,
        block_type: str,
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate content_block_start event."""
        index = self.streaming_state["current_content_block_index"]
        
        if block_type == "text":
            content_block = {
                "type": "text",
                "text": ""
            }
        elif block_type == "tool_use":
            content_block = {
                "type": "tool_use",
                "id": tool_id or f"toolu_{int(time.time() * 1000)}",
                "name": tool_name or "",
                "input": {}
            }
        else:
            content_block = {"type": block_type}
        
        return {
            "type": "content_block_start",
            "data": {
                "type": "content_block_start",
                "index": index,
                "content_block": content_block
            }
        }
    
    def _generate_text_delta_event(self, text: str) -> Dict[str, Any]:
        """Generate content_block_delta event with text_delta."""
        return {
            "type": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self.streaming_state["current_content_block_index"],
                "delta": {
                    "type": "text_delta",
                    "text": text
                }
            }
        }
    
    def _generate_input_json_delta_event(self, partial_json: str) -> Dict[str, Any]:
        """Generate content_block_delta event with input_json_delta."""
        return {
            "type": "content_block_delta",
            "data": {
                "type": "content_block_delta",
                "index": self.streaming_state["current_content_block_index"],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json
                }
            }
        }
    
    def _generate_content_block_stop_event(self) -> Dict[str, Any]:
        """Generate content_block_stop event."""
        return {
            "type": "content_block_stop",
            "data": {
                "type": "content_block_stop",
                "index": self.streaming_state["current_content_block_index"]
            }
        }
    
    def _generate_final_events(self) -> List[Dict[str, Any]]:
        """Generate final events (content_block_stop, message_delta, message_stop)."""
        events = []
        
        # Close any open content block
        if self.streaming_state["sent_content_block_start"]:
            events.append(self._generate_content_block_stop_event())
        
        # Generate message_delta with stop_reason and usage
        message_delta_data = {
            "type": "message_delta",
            "delta": {},
            "usage": self.streaming_state["usage_data"]
        }
        
        if self.streaming_state["stop_reason"]:
            message_delta_data["delta"]["stop_reason"] = self.streaming_state["stop_reason"]
        
        events.append({
            "type": "message_delta",
            "data": message_delta_data
        })
        
        # Generate message_stop
        events.append({
            "type": "message_stop",
            "data": {
                "type": "message_stop"
            }
        })
        
        return events
    
    def _generate_error_event(self, error_message: str) -> Dict[str, Any]:
        """Generate error event."""
        return {
            "type": "error",
            "data": {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error_message
                }
            }
        }
    
    def _map_stop_reason(self, openai_finish_reason: str) -> str:
        """
        Map OpenAI finish_reason to Anthropic stop_reason.
        
        Args:
            openai_finish_reason: OpenAI finish_reason value
            
        Returns:
            Anthropic stop_reason value
        """
        stop_reason_mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
            "function_call": "tool_use",
        }
        
        return stop_reason_mapping.get(openai_finish_reason, "end_turn")
    
    def format_sse_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
        """
        Format event as SSE (Server-Sent Events).
        
        SSE format:
        event: <event_type>
        data: <json_data>
        
        (blank line)
        
        Args:
            event_type: Type of SSE event
            data: Event data dictionary
            
        Returns:
            bytes: SSE-formatted event
        """
        try:
            json_data = json.dumps(data, separators=(',', ':'))
            sse_event = f"event: {event_type}\ndata: {json_data}\n\n"
            return sse_event.encode('utf-8')
        except Exception as e:
            self.logger.error(f"Error formatting SSE event: {e}")
            # Return a minimal error event
            error_data = json.dumps({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Error formatting event"
                }
            })
            return f"event: error\ndata: {error_data}\n\n".encode('utf-8')
