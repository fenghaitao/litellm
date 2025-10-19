# LiteLLM + Anthropic Integration Guide

## Overview

LiteLLM provides a unified interface to call Anthropic's Claude models using the OpenAI-compatible format. This means you can use the same code structure you'd use for OpenAI, but with Claude models.

## How It Works

### Architecture Flow

```
Your Code (OpenAI format)
    ↓
LiteLLM Translation Layer
    ↓
Anthropic API (Anthropic format)
    ↓
LiteLLM Response Normalization
    ↓
Your Code (OpenAI format response)
```

### Key Components

1. **`litellm/llms/anthropic/chat/transformation.py`**
   - Converts OpenAI-style messages to Anthropic's format
   - Handles system prompts, tools, and special parameters
   - Maps response formats between the two APIs

2. **`litellm/llms/anthropic/chat/handler.py`**
   - Makes HTTP calls to Anthropic's API
   - Handles streaming responses
   - Manages error handling and retries

3. **`litellm/llms/anthropic/common_utils.py`**
   - Utility functions for Anthropic-specific features
   - Header management (beta features, caching, etc.)
   - Token counting and cost calculation

## Supported Models

```python
# Claude 3.5 Sonnet (Recommended - Best balance)
"anthropic/claude-3-5-sonnet-20241022"

# Claude 3.5 Haiku (Fast & Affordable)
"anthropic/claude-3-5-haiku-20241022"

# Claude 3 Opus (Most Capable)
"anthropic/claude-3-opus-20240229"

# Claude Sonnet 4 (Extended Thinking)
"anthropic/claude-3-7-sonnet-20250219"
```

## Key Features

### 1. Message Translation

**Your Code (OpenAI format):**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
]
```

**What LiteLLM sends to Anthropic:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "system": "You are a helpful assistant",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 4096
}
```

### 2. Tool/Function Calling

**Your Code:**
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {...}
    }
}]
```

**LiteLLM converts to Anthropic's tool format:**
```json
{
  "tools": [{
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {...}
  }]
}
```

### 3. Streaming

LiteLLM handles Anthropic's Server-Sent Events (SSE) format and converts it to OpenAI's streaming format:

```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content)
```

### 4. Prompt Caching

Anthropic's prompt caching is supported through special content blocks:

```python
messages = [{
    "role": "system",
    "content": [{
        "type": "text",
        "text": "Large context...",
        "cache_control": {"type": "ephemeral"}  # Cache this
    }]
}]
```

LiteLLM automatically:
- Detects cache_control blocks
- Adds the required `anthropic-beta: prompt-caching-2024-07-31` header
- Returns cache hit/miss information in usage stats

### 5. Extended Thinking (Claude Sonnet 4)

```python
response = completion(
    model="anthropic/claude-3-7-sonnet-20250219",
    messages=[...],
    thinking={
        "type": "enabled",
        "budget_tokens": 5000
    }
)
```

LiteLLM handles:
- Converting the thinking parameter to Anthropic's format
- Extracting thinking blocks from the response
- Normalizing to OpenAI-compatible format

## Parameter Mapping

| OpenAI Parameter | Anthropic Parameter | Notes |
|-----------------|---------------------|-------|
| `messages` | `messages` + `system` | System messages extracted |
| `max_tokens` | `max_tokens` | Required by Anthropic (default: 4096) |
| `temperature` | `temperature` | Direct mapping |
| `top_p` | `top_p` | Direct mapping |
| `stop` | `stop_sequences` | Renamed |
| `tools` | `tools` | Format converted |
| `tool_choice` | `tool_choice` | Format converted |
| `stream` | `stream` | Direct mapping |
| `user` | `metadata.user_id` | Moved to metadata |

## Error Handling

LiteLLM normalizes Anthropic errors to OpenAI-compatible exceptions:

```python
from litellm import completion
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError
)

try:
    response = completion(...)
except RateLimitError as e:
    print(f"Rate limited: {e}")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
```

## Cost Tracking

LiteLLM automatically tracks costs using Anthropic's pricing:

```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...]
)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")

# Cost calculation (if enabled)
if hasattr(response, '_hidden_params'):
    print(f"Cost: ${response._hidden_params.get('response_cost', 0)}")
```

## Advanced Features

### 1. Vision (Image Analysis)

```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
)
```

### 2. PDF Analysis

```python
# LiteLLM automatically adds the required beta header
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this PDF"},
            {"type": "document", "source": {"type": "url", "url": "https://..."}}
        ]
    }]
)
```

### 3. Computer Use (Beta)

```python
tools = [{
    "type": "computer_20241022",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768
}]

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...],
    tools=tools
)
```

## Using with LiteLLM Proxy

You can also use Anthropic through the LiteLLM Proxy server:

### 1. Configure the proxy

**config.yaml:**
```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

### 2. Start the proxy

```bash
litellm --config config.yaml
```

### 3. Use OpenAI SDK

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="claude",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Best Practices

1. **Set max_tokens**: Anthropic requires this parameter
   ```python
   completion(model="anthropic/...", max_tokens=1024, ...)
   ```

2. **Use prompt caching for repeated contexts**: Save costs on large contexts
   ```python
   # Add cache_control to frequently reused content
   ```

3. **Handle rate limits**: Use retry logic or the Router
   ```python
   from litellm import Router
   router = Router(model_list=[...], num_retries=3)
   ```

4. **Monitor costs**: Track token usage
   ```python
   print(f"Tokens: {response.usage.total_tokens}")
   ```

5. **Use appropriate models**:
   - Haiku: Fast, cheap tasks
   - Sonnet: Balanced performance
   - Opus: Complex reasoning
   - Sonnet 4: Extended thinking tasks

## Troubleshooting

### Issue: "max_tokens is required"
**Solution:** Always set max_tokens for Anthropic
```python
completion(model="anthropic/...", max_tokens=4096, ...)
```

### Issue: Rate limit errors
**Solution:** Use Router with retries
```python
router = Router(model_list=[...], num_retries=3, retry_after=60)
```

### Issue: Tool calling not working
**Solution:** Ensure tools are in OpenAI format, LiteLLM will convert
```python
tools = [{"type": "function", "function": {...}}]
```

## Reverse Translation: Anthropic Format → Any Provider

LiteLLM Proxy also supports **reverse translation** - accepting requests in native Anthropic format and routing them to any provider (OpenAI, Azure, etc.).

### Use Case

You have existing code using Anthropic's SDK but want to:
- Route to different providers without changing code
- Add load balancing and fallbacks
- Track costs and enforce rate limits
- A/B test different models

### How It Works

```python
import anthropic

# Point Anthropic SDK to LiteLLM proxy
client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000/anthropic"  # Proxy endpoint
)

# Use native Anthropic format - routes to ANY provider!
message = client.messages.create(
    model="gpt-4",  # Can be OpenAI, Azure, etc.
    max_tokens=1024,
    system="You are helpful",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Translation Flow

```
Your Code (Anthropic SDK)
    ↓ Anthropic Format
LiteLLM Proxy (/anthropic/v1/messages)
    ↓ Translates to OpenAI Format
OpenAI API (or any provider)
    ↓ OpenAI Response
LiteLLM Proxy
    ↓ Translates back to Anthropic Format
Your Code (Anthropic SDK)
```

### Benefits

- **No code changes**: Keep using Anthropic SDK
- **Provider flexibility**: Route to OpenAI, Azure, etc.
- **Proxy features**: Fallbacks, cost tracking, rate limiting
- **Easy migration**: Switch providers without rewriting code

See [reverse_translation_example.py](./reverse_translation_example.py) for detailed examples.

## Resources

- [Anthropic API Docs](https://docs.anthropic.com/)
- [LiteLLM Docs](https://docs.litellm.ai/)
- [LiteLLM Anthropic Provider Docs](https://docs.litellm.ai/docs/providers/anthropic)
- [Example Code](./anthropic_example.py)
- [Quick Start](./anthropic_quickstart.py)
- [Reverse Translation](./reverse_translation_example.py)
