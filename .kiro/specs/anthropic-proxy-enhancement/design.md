# Design Document

## Overview

This design integrates the proven Anthropic-to-OpenAI transformation logic from `examples/anthropic/server` into the LiteLLM proxy's `/v1/messages` endpoint. The design enables Claude Code and other Anthropic SDK clients to work seamlessly with any OpenAI-compatible provider through LiteLLM's routing infrastructure.

## Architecture

### High-Level Flow

```
Anthropic Client (Claude Code)
    ↓ (Anthropic format)
LiteLLM Proxy /v1/messages endpoint
    ↓ (validate)
Anthropic Request Validator
    ↓ (transform)
Anthropic-to-OpenAI Transformer
    ↓ (OpenAI format)
LiteLLM Router / litellm.completion()
    ↓ (route to provider)
Provider (OpenAI, Azure, Anthropic, etc.)
    ↓ (OpenAI format response)
OpenAI-to-Anthropic Transformer
    ↓ (Anthropic format)
Anthropic Client (Claude Code)
```

### Component Architecture

```
litellm/proxy/anthropic_endpoints/
├── endpoints.py                    # FastAPI endpoint (existing, to be enhanced)
├── validation.py                   # Request validation logic (new)
└── transformation.py               # Transformation logic (new)

litellm/llms/anthropic/
├── chat/
│   └── transformation.py           # Existing transformation (keep for backward compat)
└── experimental_pass_through/
    └── adapters/
        ├── transformation.py       # Existing adapter (keep for backward compat)
        └── streaming_iterator.py   # Existing streaming (enhance)
```

## Components and Interfaces

### 1. Enhanced /v1/messages Endpoint

**Location:** `litellm/proxy/anthropic_endpoints/endpoints.py`

**Responsibilities:**
- Accept Anthropic-formatted requests
- Validate requests using validation module
- Transform to OpenAI format using transformation module
- Route through LiteLLM's existing infrastructure
- Transform responses back to Anthropic format
- Handle streaming with proper SSE formatting

**Interface:**
```python
@router.post("/v1/messages")
async def anthropic_response(
    fastapi_response: Response,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Accept Anthropic Messages API requests and route to any provider.
    
    Flow:
    1. Parse Anthropic request
    2. Validate request structure
    3. Transform to OpenAI format
    4. Route through litellm.completion() or router
    5. Transform response back to Anthropic format
    6. Return to client
    """
```

### 2. Request Validation Module

**Location:** `litellm/proxy/anthropic_endpoints/validation.py` (new file)

**Responsibilities:**
- Validate required fields (model, messages, max_tokens)
- Validate message structure and content blocks
- Validate tool definitions and schemas
- Validate tool_result references against tool_use blocks
- Return Anthropic-compatible error responses

**Key Functions:**
```python
def validate_anthropic_request(data: Dict[str, Any]) -> None:
    """
    Validate Anthropic Messages API request.
    Raises InvalidRequestError with Anthropic-compatible format.
    """

def validate_messages(messages: List[Dict]) -> Set[str]:
    """
    Validate message structure and content blocks.
    Returns set of valid tool_use_id values.
    """

def validate_tools(tools: Optional[List[Dict]]) -> None:
    """Validate tool definitions."""

def validate_content_blocks(
    content: Union[str, List[Dict]], 
    context: str, 
    role: str
) -> Set[str]:
    """
    Validate content blocks structure.
    Returns set of tool_use IDs found.
    """

class InvalidRequestError(Exception):
    """Anthropic-compatible validation error."""
```

### 3. Transformation Module

**Location:** `litellm/proxy/anthropic_endpoints/transformation.py` (new file)

**Responsibilities:**
- Transform Anthropic messages to OpenAI format
- Transform Anthropic tools to OpenAI format
- Transform OpenAI responses back to Anthropic format
- Handle streaming response transformation
- Map stop reasons between formats

**Key Classes and Functions:**
```python
class AnthropicToOpenAITransformer:
    """Transform Anthropic requests to OpenAI format."""
    
    def transform_messages(
        self,
        messages: List[Dict],
        system: Optional[Union[str, List[Dict]]] = None
    ) -> List[Dict]:
        """
        Transform Anthropic messages to OpenAI format.
        
        Handles:
        - Text content blocks → string content
        - tool_use blocks → tool_calls
        - tool_result blocks → tool messages
        - image blocks → image_url format
        - system parameter → system message
        """
    
    def transform_tools(
        self,
        tools: List[Dict]
    ) -> List[Dict]:
        """
        Transform Anthropic tools to OpenAI format.
        
        Converts:
        - input_schema → parameters
        - Anthropic tool format → OpenAI function format
        """
    
    def transform_tool_choice(
        self,
        tool_choice: Union[str, Dict]
    ) -> Union[str, Dict]:
        """
        Transform Anthropic tool_choice to OpenAI format.
        
        Mappings:
        - "auto" → "auto"
        - "any" → "required"
        - {"type": "tool", "name": "x"} → {"type": "function", "function": {"name": "x"}}
        """

class OpenAIToAnthropicTransformer:
    """Transform OpenAI responses to Anthropic format."""
    
    def transform_response(
        self,
        openai_response: Dict,
        original_model: str
    ) -> Dict:
        """
        Transform OpenAI response to Anthropic format.
        
        Converts:
        - text content → text content block
        - tool_calls → tool_use content blocks
        - finish_reason → stop_reason
        - usage → Anthropic usage format
        """
    
    def transform_streaming_chunk(
        self,
        openai_chunk: Dict,
        streaming_state: Dict
    ) -> List[Dict]:
        """
        Transform OpenAI streaming chunk to Anthropic SSE events.
        
        Returns list of events:
        - message_start
        - content_block_start
        - content_block_delta
        - content_block_stop
        - message_delta
        - message_stop
        """
    
    def map_stop_reason(
        self,
        openai_finish_reason: str
    ) -> str:
        """
        Map OpenAI finish_reason to Anthropic stop_reason.
        
        Mappings:
        - "stop" → "end_turn"
        - "length" → "max_tokens"
        - "tool_calls" → "tool_use"
        - "content_filter" → "stop_sequence"
        """
```

### 4. Streaming Handler Enhancement

**Location:** `litellm/proxy/anthropic_endpoints/streaming.py` (new file)

**Responsibilities:**
- Convert OpenAI streaming chunks to Anthropic SSE events
- Maintain streaming state across chunks
- Handle content block transitions
- Merge usage and stop_reason events
- Ensure proper event sequencing

**Key Classes:**
```python
class AnthropicStreamingHandler:
    """Handle streaming response transformation."""
    
    def __init__(self):
        self.streaming_state = {
            "message_id": None,
            "current_content_block_index": 0,
            "current_block_type": "text",
            "sent_message_start": False,
            "sent_content_block_start": False,
            "tool_use_ids": set(),
        }
    
    async def transform_stream(
        self,
        openai_stream: AsyncIterator,
        model: str
    ) -> AsyncIterator[bytes]:
        """
        Transform OpenAI stream to Anthropic SSE format.
        
        Yields SSE-formatted events:
        event: message_start
        data: {...}
        
        event: content_block_delta
        data: {...}
        """
    
    def format_sse_event(
        self,
        event_type: str,
        data: Dict
    ) -> bytes:
        """Format event as SSE."""
```

## Data Models

### Anthropic Request Format
```python
{
    "model": "claude-3-sonnet-20240229",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "source": {...}}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hi"},
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {...}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "..."}
            ]
        }
    ],
    "system": "You are a helpful assistant",
    "tools": [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {...}
            }
        }
    ],
    "tool_choice": {"type": "auto"},
    "max_tokens": 1024,
    "temperature": 1.0,
    "stream": false
}
```

### Transformed OpenAI Format
```python
{
    "model": "gpt-4",  # or any provider model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi", "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{...}"
                }
            }
        ]},
        {"role": "tool", "tool_call_id": "call_123", "content": "..."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {...}
                }
            }
        }
    ],
    "tool_choice": "auto",
    "max_tokens": 1024,
    "temperature": 1.0,
    "stream": false
}
```

### Anthropic Response Format
```python
{
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The weather is..."},
        {"type": "tool_use", "id": "toolu_456", "name": "get_weather", "input": {...}}
    ],
    "model": "claude-3-sonnet-20240229",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50
    }
}
```

### Anthropic Streaming Events
```
event: message_start
data: {"type": "message_start", "message": {...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}

event: message_stop
data: {"type": "message_stop"}
```

## Error Handling

### Error Response Format
All errors should follow Anthropic's error format:

```python
{
    "type": "error",
    "error": {
        "type": "invalid_request_error",  # or authentication_error, rate_limit_error, api_error
        "message": "Detailed error message"
    }
}
```

### Error Types
- `invalid_request_error` - Validation failures (400)
- `authentication_error` - Auth failures (401)
- `permission_error` - Permission issues (403)
- `not_found_error` - Resource not found (404)
- `rate_limit_error` - Rate limit exceeded (429)
- `api_error` - Provider API errors (500)
- `overloaded_error` - System overloaded (529)

### Validation Error Examples
```python
# Missing required field
{
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "model is required"
    }
}

# Invalid tool_result reference
{
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "tool_result references invalid tool_use_id 'toolu_123'. Valid IDs: ['toolu_456']"
    }
}

# Invalid content block
{
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": "Message at index 0, content block 1: text blocks must have a 'text' field"
    }
}
```

## Testing Strategy

### Unit Tests
1. **Validation Tests** (`test_anthropic_validation.py`)
   - Test required field validation
   - Test message structure validation
   - Test tool definition validation
   - Test content block validation
   - Test tool_result reference validation

2. **Transformation Tests** (`test_anthropic_transformation.py`)
   - Test message transformation (Anthropic → OpenAI)
   - Test tool transformation
   - Test response transformation (OpenAI → Anthropic)
   - Test stop_reason mapping
   - Test edge cases (empty content, nested structures)

3. **Streaming Tests** (`test_anthropic_streaming.py`)
   - Test SSE event formatting
   - Test event sequencing
   - Test content block transitions
   - Test usage/stop_reason merging
   - Test error handling in streams

### Integration Tests
1. **End-to-End Tests** (`test_anthropic_e2e.py`)
   - Test with real Anthropic SDK client
   - Test with Claude Code scenarios
   - Test with multiple providers (OpenAI, Azure, Anthropic)
   - Test streaming responses
   - Test tool use workflows

2. **Router Integration Tests** (`test_anthropic_router.py`)
   - Test with LiteLLM router
   - Test load balancing
   - Test fallbacks
   - Test rate limiting
   - Test logging

### Compatibility Tests
1. **Backward Compatibility** (`test_backward_compat.py`)
   - Ensure existing `/v1/messages` behavior works
   - Ensure existing pass-through mode works
   - Ensure existing OpenAI endpoints unaffected

## Implementation Phases

### Phase 1: Core Transformation Logic
- Create validation module with comprehensive checks
- Create transformation module with Anthropic ↔ OpenAI conversion
- Add unit tests for validation and transformation
- Ensure all edge cases are handled

### Phase 2: Endpoint Integration
- Enhance `/v1/messages` endpoint to use new transformation
- Integrate with existing LiteLLM routing
- Add error handling with Anthropic format
- Add logging and monitoring

### Phase 3: Streaming Support
- Create streaming handler for SSE formatting
- Implement proper event sequencing
- Handle content block transitions
- Add streaming tests

### Phase 4: Testing and Validation
- Add comprehensive integration tests
- Test with real Anthropic SDK
- Test with Claude Code
- Test with multiple providers
- Performance testing

### Phase 5: Documentation and Deployment
- Update documentation
- Add configuration examples
- Create migration guide
- Deploy to production

## Configuration

### Proxy Configuration
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: gpt-4  # Route to OpenAI
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-opus-20240229
    litellm_params:
      model: azure/gpt-4  # Route to Azure
      api_base: https://...
      api_key: os.environ/AZURE_API_KEY

# Enable Anthropic transformation mode
anthropic_transformation_enabled: true
```

### Environment Variables
```bash
# Provider API keys
OPENAI_API_KEY=sk-...
AZURE_API_KEY=...
ANTHROPIC_API_KEY=...

# LiteLLM configuration
LITELLM_LOG=DEBUG
```

## Security Considerations

1. **Input Validation**
   - Validate all input fields before transformation
   - Sanitize content blocks
   - Validate tool schemas
   - Prevent injection attacks

2. **API Key Handling**
   - Use existing LiteLLM auth mechanisms
   - Don't expose provider keys in responses
   - Log auth failures appropriately

3. **Rate Limiting**
   - Use existing LiteLLM rate limiting
   - Apply limits per user/key
   - Return proper Anthropic error format

4. **Error Information**
   - Don't expose internal implementation details
   - Sanitize error messages
   - Log detailed errors server-side only

## Performance Considerations

1. **Transformation Overhead**
   - Minimize object copying
   - Use efficient JSON parsing
   - Cache compiled regex patterns
   - Profile transformation performance

2. **Streaming Performance**
   - Use async iterators efficiently
   - Minimize buffering
   - Stream events as soon as available
   - Monitor memory usage

3. **Validation Performance**
   - Validate only once per request
   - Use early returns for invalid data
   - Cache validation results where possible

## Monitoring and Observability

1. **Metrics**
   - Request count by endpoint
   - Transformation time
   - Validation failures
   - Provider routing distribution
   - Error rates by type

2. **Logging**
   - Log all validation failures
   - Log transformation errors
   - Log provider routing decisions
   - Log performance metrics

3. **Tracing**
   - Trace request through transformation
   - Trace provider routing
   - Trace response transformation
   - Correlate with existing LiteLLM traces

## Migration Path

### For Existing Users
1. **No Breaking Changes**
   - Existing `/v1/messages` behavior preserved
   - Existing pass-through mode still works
   - OpenAI endpoints unaffected

2. **Opt-In Enhancement**
   - New transformation enabled by default for new deployments
   - Existing deployments can opt-in via configuration
   - Gradual rollout supported

### For New Users
1. **Default Behavior**
   - Anthropic transformation enabled by default
   - Works with any provider out of the box
   - Clear documentation and examples

## Success Criteria

1. **Functionality**
   - Claude Code works without modifications
   - All Anthropic SDK features supported
   - Works with all LiteLLM providers
   - Streaming works correctly

2. **Performance**
   - Transformation overhead < 10ms
   - Streaming latency < 50ms
   - No memory leaks
   - Handles high throughput

3. **Reliability**
   - 99.9% success rate for valid requests
   - Clear error messages for invalid requests
   - Graceful degradation on provider failures
   - No data loss in streaming

4. **Compatibility**
   - 100% backward compatibility
   - Works with existing LiteLLM features
   - Integrates with router, logging, auth
   - No breaking changes
