# LiteLLM + Anthropic: Code Flow Explanation

## How a Request Flows Through LiteLLM

This document explains exactly what happens when you call `completion(model="anthropic/claude-3-5-sonnet-20241022", messages=[...])`.

## Step-by-Step Flow

### 1. Your Code
```python
from litellm import completion

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=1024
)
```

### 2. Entry Point: `litellm/main.py`

The `completion()` function is the main entry point:

```python
# litellm/main.py (simplified)
def completion(
    model: str,
    messages: List,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
):
    # 1. Determine the provider from model name
    model, custom_llm_provider, _, _ = get_llm_provider(
        model=model  # "anthropic/claude-3-5-sonnet-20241022"
    )
    # Returns: model="claude-3-5-sonnet-20241022", provider="anthropic"
    
    # 2. Route to the appropriate handler
    if custom_llm_provider == "anthropic":
        return anthropic_chat_completions.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
```

### 3. Anthropic Handler: `litellm/llms/anthropic/chat/handler.py`

```python
# litellm/llms/anthropic/chat/handler.py (simplified)
class AnthropicChatCompletion(BaseLLM):
    def completion(
        self,
        model: str,
        messages: List,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: str,
        logging_obj,
        optional_params: dict,
        acompletion: bool,
        litellm_params: dict,
        logger_fn=None,
        headers: dict = {},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
    ):
        # 1. Get API key from environment
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # 2. Set API base URL
        api_base = api_base or "https://api.anthropic.com/v1/messages"
        
        # 3. Transform request to Anthropic format
        data = self.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            headers=headers
        )
        
        # 4. Make HTTP request
        if acompletion:
            return self.async_completion(...)
        else:
            return self.sync_completion(...)
```

### 4. Request Transformation: `litellm/llms/anthropic/chat/transformation.py`

This is where the magic happens - converting OpenAI format to Anthropic format:

```python
# litellm/llms/anthropic/chat/transformation.py (simplified)
class AnthropicConfig(BaseConfig):
    def transform_request(
        self,
        model: str,
        messages: List,
        optional_params: dict,
        headers: dict
    ) -> dict:
        # 1. Extract system message
        system_message = None
        user_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                user_messages.append(message)
        
        # 2. Build Anthropic request
        data = {
            "model": model,  # "claude-3-5-sonnet-20241022"
            "messages": user_messages,
            "max_tokens": optional_params.get("max_tokens", 4096)
        }
        
        # 3. Add system message if present
        if system_message:
            data["system"] = system_message
        
        # 4. Add optional parameters
        if "temperature" in optional_params:
            data["temperature"] = optional_params["temperature"]
        
        if "top_p" in optional_params:
            data["top_p"] = optional_params["top_p"]
        
        if "stop" in optional_params:
            data["stop_sequences"] = optional_params["stop"]
        
        # 5. Handle tools (function calling)
        if "tools" in optional_params:
            data["tools"] = self.transform_tools(optional_params["tools"])
        
        # 6. Set headers
        headers = self.get_anthropic_headers(
            api_key=api_key,
            prompt_caching_set=self.has_cache_control(messages),
            computer_tool_used=self.has_computer_tool(optional_params.get("tools")),
            pdf_used=self.has_pdf(messages)
        )
        
        return data, headers
    
    def transform_tools(self, openai_tools: List) -> List:
        """Convert OpenAI tool format to Anthropic format"""
        anthropic_tools = []
        
        for tool in openai_tools:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"]
                })
        
        return anthropic_tools
```

### 5. HTTP Request

The transformed data is sent to Anthropic:

```python
# What gets sent to Anthropic API
POST https://api.anthropic.com/v1/messages
Headers:
  x-api-key: sk-ant-...
  anthropic-version: 2023-06-01
  content-type: application/json

Body:
{
  "model": "claude-3-5-sonnet-20241022",
  "system": "You are helpful",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 1024,
  "temperature": 0.7
}
```

### 6. Response from Anthropic

Anthropic returns:

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! I'm Claude, an AI assistant..."
    }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 15,
    "output_tokens": 25
  }
}
```

### 7. Response Transformation

LiteLLM converts this to OpenAI format:

```python
# litellm/llms/anthropic/chat/transformation.py
def transform_response(anthropic_response: dict) -> ModelResponse:
    """Convert Anthropic response to OpenAI format"""
    
    # Extract text content
    content = ""
    for block in anthropic_response["content"]:
        if block["type"] == "text":
            content += block["text"]
    
    # Build OpenAI-compatible response
    return ModelResponse(
        id=anthropic_response["id"],
        created=int(time.time()),
        model=anthropic_response["model"],
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": map_finish_reason(
                    anthropic_response["stop_reason"]
                )
            }
        ],
        usage={
            "prompt_tokens": anthropic_response["usage"]["input_tokens"],
            "completion_tokens": anthropic_response["usage"]["output_tokens"],
            "total_tokens": (
                anthropic_response["usage"]["input_tokens"] +
                anthropic_response["usage"]["output_tokens"]
            )
        }
    )
```

### 8. Your Code Receives

```python
# What you get back (OpenAI format)
ModelResponse(
    id="msg_01XFDUDYJgAACzvnptvVoYEL",
    created=1234567890,
    model="claude-3-5-sonnet-20241022",
    object="chat.completion",
    choices=[
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm Claude, an AI assistant..."
            },
            "finish_reason": "stop"
        }
    ],
    usage={
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40
    }
)
```

## Streaming Flow

For streaming requests, the flow is similar but with additional handling:

### 1. Your Code
```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...],
    stream=True  # Enable streaming
)

for chunk in response:
    print(chunk.choices[0].delta.content)
```

### 2. Anthropic Sends SSE (Server-Sent Events)

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_123",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,...}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}
```

### 3. LiteLLM Converts to OpenAI Streaming Format

```python
# litellm/llms/anthropic/chat/handler.py
class ModelResponseIterator:
    def __iter__(self):
        for line in self.streaming_response:
            # Parse SSE event
            if line.startswith("data: "):
                data = json.loads(line[6:])
                
                if data["type"] == "content_block_delta":
                    # Convert to OpenAI chunk format
                    yield ModelResponseStream(
                        id="msg_123",
                        created=int(time.time()),
                        model="claude-3-5-sonnet-20241022",
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {
                                "content": data["delta"]["text"]
                            },
                            "finish_reason": None
                        }]
                    )
```

### 4. You Receive OpenAI-Format Chunks

```python
# Each iteration gives you:
ModelResponseStream(
    id="msg_123",
    object="chat.completion.chunk",
    choices=[{
        "delta": {"content": "Hello"},
        "finish_reason": None
    }]
)
```

## Key Takeaways

1. **Transparent Translation**: You write OpenAI-format code, LiteLLM handles Anthropic's format
2. **Provider Detection**: Model name prefix (`anthropic/`) determines the provider
3. **Request Transformation**: OpenAI params → Anthropic params
4. **Response Normalization**: Anthropic response → OpenAI format
5. **Streaming Conversion**: SSE events → OpenAI chunks
6. **Error Mapping**: Anthropic errors → OpenAI-compatible exceptions

## File Reference

Key files involved in the flow:

```
litellm/
├── main.py                                    # Entry point
├── llms/
│   └── anthropic/
│       ├── chat/
│       │   ├── handler.py                    # HTTP calls
│       │   └── transformation.py             # Format conversion
│       ├── common_utils.py                   # Utilities
│       └── cost_calculation.py               # Cost tracking
└── types/
    └── llms/
        └── anthropic.py                       # Type definitions
```

This architecture allows LiteLLM to support 100+ providers with the same interface!
