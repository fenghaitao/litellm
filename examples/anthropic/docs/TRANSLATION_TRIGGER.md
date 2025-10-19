# When is `translate_anthropic_messages_to_openai` Triggered?

## Quick Answer

`translate_anthropic_messages_to_openai` is triggered when:

1. You send a request to `/v1/messages` (Anthropic format endpoint)
2. With a model name that is **NOT** an Anthropic model (e.g., `gpt-4`, `azure/gpt-4`)

## Complete Flow with Code

### Scenario: Using Anthropic SDK with OpenAI Model

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Request with NON-Anthropic model
message = client.messages.create(
    model="gpt-4",  # ← This triggers translation!
    max_tokens=1024,
    system="You are helpful",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Step-by-Step Code Execution

#### Step 1: Request Arrives at Proxy

```python
# litellm/proxy/anthropic_endpoints/endpoints.py

@router.post("/v1/messages")
async def anthropic_response(request: Request):
    # Receive Anthropic format request
    data = {
        "model": "gpt-4",
        "max_tokens": 1024,
        "system": "You are helpful",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    # Route to handler
    llm_coro = llm_router.aanthropic_messages(**data)
```

#### Step 2: Router Calls anthropic_messages

```python
# litellm/router.py

class Router:
    def aanthropic_messages(self, **kwargs):
        # Calls litellm.anthropic_messages
        return litellm.anthropic_messages(**kwargs)
```

#### Step 3: Determine Provider

```python
# litellm/llms/anthropic/experimental_pass_through/messages/handler.py

def anthropic_messages_handler(
    model="gpt-4",
    messages=[...],
    system="You are helpful",
    max_tokens=1024,
    **kwargs
):
    # Extract provider from model name
    custom_llm_provider = get_provider_from_model("gpt-4")
    # Result: custom_llm_provider = "openai"
    
    # Try to get Anthropic-specific config
    anthropic_messages_provider_config = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model="gpt-4",
            provider="openai"  # ← OpenAI provider
        )
    )
    # Result: anthropic_messages_provider_config = None
    #         (because OpenAI doesn't have Anthropic message format)
    
    # Decision point
    if anthropic_messages_provider_config is None:
        # ← THIS BRANCH IS TAKEN!
        # No Anthropic config → Translation required
        return LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            system=system,
            **kwargs
        )
```

#### Step 4: Translation Handler Called

```python
# litellm/llms/anthropic/experimental_pass_through/adapters/handler.py

class LiteLLMMessagesToCompletionTransformationHandler:
    @staticmethod
    def anthropic_messages_handler(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        system="You are helpful",
        max_tokens=1024,
        **kwargs
    ):
        # Prepare request data
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful",
            "max_tokens": 1024
        }
        
        # ← TRANSLATION HAPPENS HERE!
        openai_request = ANTHROPIC_ADAPTER.translate_completion_input_params(
            request_data
        )
        
        # Call litellm.completion with OpenAI format
        completion_response = litellm.completion(**openai_request)
        
        # Translate response back to Anthropic format
        anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(
            completion_response
        )
        
        return anthropic_response
```

#### Step 5: translate_completion_input_params

```python
# litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py

class AnthropicAdapter:
    def translate_completion_input_params(self, kwargs):
        """
        Translate Anthropic format → OpenAI format
        """
        # Input (Anthropic format):
        # {
        #     "model": "gpt-4",
        #     "messages": [{"role": "user", "content": "Hello"}],
        #     "system": "You are helpful",
        #     "max_tokens": 1024
        # }
        
        model = kwargs.pop("model")  # "gpt-4"
        messages = kwargs.pop("messages")  # [{"role": "user", ...}]
        
        # Create Anthropic request object
        request_body = AnthropicMessagesRequest(
            model=model,
            messages=messages,
            **kwargs  # includes system, max_tokens, etc.
        )
        
        # ← ACTUAL TRANSLATION HERE!
        translated_body = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            anthropic_message_request=request_body
        )
        
        # Output (OpenAI format):
        # {
        #     "model": "gpt-4",
        #     "messages": [
        #         {"role": "system", "content": "You are helpful"},
        #         {"role": "user", "content": "Hello"}
        #     ],
        #     "max_tokens": 1024
        # }
        
        return translated_body
```

#### Step 6: translate_anthropic_to_openai

```python
# litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py

class LiteLLMAnthropicMessagesAdapter:
    def translate_anthropic_to_openai(
        self,
        anthropic_message_request: AnthropicMessagesRequest
    ) -> ChatCompletionRequest:
        """
        THE ACTUAL TRANSLATION METHOD!
        
        Converts Anthropic format to OpenAI format
        """
        
        # Extract Anthropic-specific fields
        system = anthropic_message_request.get("system")
        messages = anthropic_message_request.get("messages")
        tools = anthropic_message_request.get("tools")
        
        # Build OpenAI messages
        openai_messages = []
        
        # 1. Convert system message
        if system:
            openai_messages.append({
                "role": "system",
                "content": system
            })
        
        # 2. Add user/assistant messages
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 3. Convert tools (if any)
        openai_tools = None
        if tools:
            openai_tools = self._translate_tools_anthropic_to_openai(tools)
        
        # 4. Build OpenAI request
        return ChatCompletionRequest(
            model=anthropic_message_request["model"],
            messages=openai_messages,
            tools=openai_tools,
            max_tokens=anthropic_message_request["max_tokens"],
            temperature=anthropic_message_request.get("temperature"),
            top_p=anthropic_message_request.get("top_p"),
            # ... other params
        )
```

#### Step 7: Call OpenAI API

```python
# Back in handler.py

# Now we have OpenAI format request
openai_request = {
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024
}

# Call litellm.completion (which calls OpenAI)
completion_response = litellm.completion(**openai_request)

# Response from OpenAI (OpenAI format):
# {
#     "id": "chatcmpl-123",
#     "object": "chat.completion",
#     "choices": [{
#         "message": {
#             "role": "assistant",
#             "content": "Hello! How can I help you?"
#         }
#     }],
#     "usage": {"prompt_tokens": 10, "completion_tokens": 8}
# }
```

#### Step 8: Translate Response Back

```python
# litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py

class AnthropicAdapter:
    def translate_completion_output_params(
        self,
        completion_response: ModelResponse
    ) -> AnthropicMessagesResponse:
        """
        Translate OpenAI response → Anthropic format
        """
        
        # Input (OpenAI format):
        # {
        #     "id": "chatcmpl-123",
        #     "choices": [{
        #         "message": {
        #             "role": "assistant",
        #             "content": "Hello! How can I help you?"
        #         }
        #     }],
        #     "usage": {"prompt_tokens": 10, "completion_tokens": 8}
        # }
        
        # Extract content
        content = completion_response.choices[0].message.content
        
        # Build Anthropic response
        return AnthropicMessagesResponse(
            id=completion_response.id,
            type="message",
            role="assistant",
            content=[{
                "type": "text",
                "text": content
            }],
            model=completion_response.model,
            usage={
                "input_tokens": completion_response.usage.prompt_tokens,
                "output_tokens": completion_response.usage.completion_tokens
            },
            stop_reason="end_turn"
        )
        
        # Output (Anthropic format):
        # {
        #     "id": "chatcmpl-123",
        #     "type": "message",
        #     "role": "assistant",
        #     "content": [{
        #         "type": "text",
        #         "text": "Hello! How can I help you?"
        #     }],
        #     "model": "gpt-4",
        #     "usage": {
        #         "input_tokens": 10,
        #         "output_tokens": 8
        #     }
        # }
```

#### Step 9: Return to Client

```python
# Your code receives Anthropic format response
message = client.messages.create(...)

print(message.content[0].text)
# Output: "Hello! How can I help you?"

print(message.usage.input_tokens)
# Output: 10
```

## When Translation is NOT Triggered

### Scenario: Using Anthropic Model

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Request with Anthropic model
message = client.messages.create(
    model="anthropic/claude-3-5-sonnet-20241022",  # ← Anthropic model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

**Flow:**

```python
# Step 3: Determine Provider
custom_llm_provider = "anthropic"  # ← Anthropic provider

anthropic_messages_provider_config = get_provider_anthropic_messages_config(
    model="anthropic/claude-3-5-sonnet-20241022",
    provider="anthropic"
)
# Result: anthropic_messages_provider_config = AnthropicMessagesConfig()
#         (NOT None!)

# Decision point
if anthropic_messages_provider_config is None:
    # Translation path
    pass
else:
    # ← THIS BRANCH IS TAKEN!
    # Pass-through path (no translation)
    return base_llm_http_handler.anthropic_messages_handler(...)
```

## Summary Table

| Model Name | Provider | Config Found? | Translation? | Path |
|------------|----------|---------------|--------------|------|
| `gpt-4` | openai | ❌ No | ✅ Yes | Adapter |
| `openai/gpt-4` | openai | ❌ No | ✅ Yes | Adapter |
| `azure/gpt-4` | azure | ❌ No | ✅ Yes | Adapter |
| `anthropic/claude-3-5-sonnet` | anthropic | ✅ Yes | ❌ No | Pass-through |
| `claude-3-5-sonnet` (if configured) | anthropic | ✅ Yes | ❌ No | Pass-through |
| `bedrock/anthropic.claude-v2` | bedrock | ❌ No | ✅ Yes | Adapter |
| `vertex_ai/claude-3-sonnet` | vertex_ai | ✅ Yes* | ❌ No | Pass-through |

*Vertex AI with Claude models has Anthropic message support

## Key Methods and Their Roles

### 1. `anthropic_messages_handler` (messages/handler.py)
**Role:** Decision point - pass-through or translate?
```python
if anthropic_messages_provider_config is None:
    # Translate
    return adapter.translate(...)
else:
    # Pass-through
    return pass_through(...)
```

### 2. `LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler` (adapters/handler.py)
**Role:** Orchestrates the translation process
```python
# 1. Translate request
openai_request = ANTHROPIC_ADAPTER.translate_completion_input_params(...)

# 2. Call OpenAI
response = litellm.completion(**openai_request)

# 3. Translate response back
anthropic_response = ANTHROPIC_ADAPTER.translate_completion_output_params(response)
```

### 3. `translate_completion_input_params` (adapters/transformation.py)
**Role:** Entry point for request translation
```python
def translate_completion_input_params(self, kwargs):
    request_body = AnthropicMessagesRequest(**kwargs)
    return self.translate_anthropic_to_openai(request_body)
```

### 4. `translate_anthropic_to_openai` (adapters/transformation.py)
**Role:** THE ACTUAL TRANSLATION - Anthropic → OpenAI
```python
def translate_anthropic_to_openai(self, anthropic_request):
    # Convert system message
    # Convert messages
    # Convert tools
    # Build OpenAI request
    return ChatCompletionRequest(...)
```

### 5. `translate_completion_output_params` (adapters/transformation.py)
**Role:** Translate response back - OpenAI → Anthropic
```python
def translate_completion_output_params(self, openai_response):
    # Extract content
    # Build Anthropic response
    return AnthropicMessagesResponse(...)
```

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Your Code: client.messages.create(model="gpt-4", ...)      │
└────────────────────────┬────────────────────────────────────┘
                         │ Anthropic format
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Proxy: /v1/messages endpoint                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ anthropic_messages_handler                                  │
│ • Extract provider from model: "gpt-4" → "openai"          │
│ • Check for Anthropic config: None found                   │
│ • Decision: Translation required                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ LiteLLMMessagesToCompletionTransformationHandler            │
│ .anthropic_messages_handler                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ANTHROPIC_ADAPTER.translate_completion_input_params         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ translate_anthropic_to_openai ← YOU ARE HERE!               │
│ • system → messages[0] with role="system"                   │
│ • Convert tools format                                      │
│ • Build OpenAI request                                      │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI format
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ litellm.completion(model="gpt-4", messages=[...])           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ OpenAI API                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │ OpenAI response
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ translate_completion_output_params                          │
│ • Convert OpenAI response → Anthropic format                │
└────────────────────────┬────────────────────────────────────┘
                         │ Anthropic format
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Your Code: message.content[0].text                          │
└─────────────────────────────────────────────────────────────┘
```

## Testing Translation

### Test that Translation is Triggered

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# This SHOULD trigger translation
message = client.messages.create(
    model="gpt-4",  # Non-Anthropic model
    max_tokens=50,
    system="You are a helpful assistant",
    messages=[{"role": "user", "content": "Say 'translation works'"}]
)

print(message.content[0].text)
# Should contain "translation works" or similar
# Response is in Anthropic format even though it came from OpenAI

# Check the response structure
assert message.type == "message"
assert message.role == "assistant"
assert isinstance(message.content, list)
assert message.content[0]["type"] == "text"
```

### Check Proxy Logs

When translation is triggered, you'll see in proxy logs:

```
[INFO] Received request to /v1/messages
[INFO] Model: gpt-4
[INFO] Provider: openai
[INFO] No Anthropic config found - using adapter
[INFO] Translating Anthropic → OpenAI format
[INFO] Calling litellm.completion with OpenAI format
[INFO] Translating OpenAI → Anthropic format
[INFO] Returning Anthropic format response
```

## Conclusion

**`translate_anthropic_to_openai` is triggered when:**

1. Request arrives at `/v1/messages` (Anthropic endpoint)
2. Model name indicates a non-Anthropic provider (e.g., `gpt-4`)
3. No Anthropic-specific config is found for that provider
4. The adapter is invoked to handle the translation

**The translation allows you to:**
- Use Anthropic SDK with any LLM provider
- Keep your code format unchanged
- Switch providers by changing model name only
- Get responses in Anthropic format regardless of underlying provider
