# How LiteLLM Decides: Pass-Through vs Translation

## Quick Answer

LiteLLM decides based on the **model name** in your request:

- **Model starts with `anthropic/`** → Pass-through to Anthropic API (no translation)
- **Model is anything else** (e.g., `gpt-4`, `azure/gpt-4`) → Translate to OpenAI format

## Decision Flow

```
Request arrives at /v1/messages (Anthropic format)
    ↓
Extract model name from request
    ↓
    ├─ Model = "anthropic/claude-3-5-sonnet" ?
    │   ↓ YES
    │   Pass-through to Anthropic API
    │   (No translation needed)
    │
    └─ Model = "gpt-4" or "azure/gpt-4" ?
        ↓ YES
        Translate Anthropic → OpenAI format
        Send to OpenAI/Azure API
        Translate response back
```

## Code Flow

### Step 1: Request Arrives

```python
# Your code
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="gpt-4",  # ← This determines the routing!
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Step 2: Proxy Receives Request

```python
# litellm/proxy/anthropic_endpoints/endpoints.py

@router.post("/v1/messages")
async def anthropic_response(request: Request):
    data = await request.json()
    
    # Extract model name
    model = data["model"]  # "gpt-4"
    
    # Route to appropriate handler
    if llm_router is not None:
        llm_coro = llm_router.aanthropic_messages(**data)
    else:
        llm_coro = litellm.anthropic_messages(**data)
```

### Step 3: Determine Provider

```python
# litellm/llms/anthropic/experimental_pass_through/messages/handler.py

def anthropic_messages_handler(model, messages, **kwargs):
    # Determine provider from model name
    custom_llm_provider = get_provider_from_model(model)
    # "gpt-4" → custom_llm_provider = "openai"
    # "anthropic/claude-3-5-sonnet" → custom_llm_provider = "anthropic"
    
    # Get provider config
    anthropic_messages_provider_config = (
        ProviderConfigManager.get_provider_anthropic_messages_config(
            model=model,
            provider=custom_llm_provider
        )
    )
    
    # Decision point
    if anthropic_messages_provider_config is None:
        # No Anthropic-specific config found
        # → This is NOT an Anthropic model
        # → Use adapter to translate
        return LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(...)
    else:
        # Anthropic-specific config found
        # → This IS an Anthropic model
        # → Pass through without translation
        return base_llm_http_handler.anthropic_messages_handler(...)
```

### Step 4A: Pass-Through (Anthropic Model)

```python
# When model = "anthropic/claude-3-5-sonnet"

# litellm/llms/anthropic/experimental_pass_through/messages/transformation.py

class AnthropicMessagesConfig:
    def transform_anthropic_messages_request(self, model, messages, **kwargs):
        """
        No transformation needed - already in Anthropic format!
        """
        return AnthropicMessagesRequest(
            model=model,
            messages=messages,
            **kwargs
        )
    
    def transform_anthropic_messages_response(self, raw_response):
        """
        No transformation needed - return as-is!
        """
        return AnthropicMessagesResponse(**raw_response.json())
```

**Flow:**
```
Your Request (Anthropic format)
    ↓
LiteLLM Proxy
    ↓ No translation
Anthropic API
    ↓ Anthropic response
LiteLLM Proxy
    ↓ No translation
Your Code (Anthropic format)
```

### Step 4B: Translation (Non-Anthropic Model)

```python
# When model = "gpt-4"

# litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py

class AnthropicAdapter:
    def translate_anthropic_to_openai(self, anthropic_request):
        """
        Translate Anthropic format → OpenAI format
        """
        # Extract system message
        system_message = anthropic_request.get("system")
        
        # Build OpenAI messages
        openai_messages = []
        if system_message:
            openai_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add user/assistant messages
        openai_messages.extend(anthropic_request["messages"])
        
        # Translate tools
        openai_tools = self.translate_tools(anthropic_request.get("tools"))
        
        return ChatCompletionRequest(
            model=anthropic_request["model"],
            messages=openai_messages,
            tools=openai_tools,
            max_tokens=anthropic_request["max_tokens"],
            temperature=anthropic_request.get("temperature"),
            # ... other params
        )
    
    def translate_openai_to_anthropic(self, openai_response):
        """
        Translate OpenAI response → Anthropic format
        """
        return AnthropicMessagesResponse(
            id=openai_response.id,
            type="message",
            role="assistant",
            content=[{
                "type": "text",
                "text": openai_response.choices[0].message.content
            }],
            model=openai_response.model,
            usage={
                "input_tokens": openai_response.usage.prompt_tokens,
                "output_tokens": openai_response.usage.completion_tokens
            }
        )
```

**Flow:**
```
Your Request (Anthropic format)
    ↓
LiteLLM Proxy
    ↓ Translate: Anthropic → OpenAI
OpenAI API
    ↓ OpenAI response
LiteLLM Proxy
    ↓ Translate: OpenAI → Anthropic
Your Code (Anthropic format)
```

## Model Name Patterns

### Pass-Through (No Translation)

These model names trigger pass-through to Anthropic:

```python
# All these go directly to Anthropic API
"anthropic/claude-3-5-sonnet-20241022"
"anthropic/claude-3-5-haiku-20241022"
"anthropic/claude-3-opus-20240229"
"anthropic/claude-3-7-sonnet-20250219"

# Or if configured in config.yaml with litellm_params.model starting with "anthropic/"
"claude"  # If mapped to "anthropic/claude-3-5-sonnet"
```

### Translation Required

These model names trigger translation:

```python
# OpenAI models
"gpt-4"
"gpt-4-turbo"
"gpt-3.5-turbo"
"openai/gpt-4"

# Azure models
"azure/gpt-4"
"azure/gpt-35-turbo"

# Other providers
"bedrock/anthropic.claude-v2"
"vertex_ai/claude-3-sonnet"
"cohere/command-r-plus"

# Any model not starting with "anthropic/"
```

## Configuration Examples

### Example 1: Pass-Through to Anthropic

```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022  # ← Starts with "anthropic/"
      api_key: os.environ/ANTHROPIC_API_KEY
```

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Uses model name from config
message = client.messages.create(
    model="claude",  # Resolves to "anthropic/claude-3-5-sonnet-20241022"
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Flow: Anthropic format → Pass-through → Anthropic API
```

### Example 2: Translation to OpenAI

```yaml
# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4  # ← Does NOT start with "anthropic/"
      api_key: os.environ/OPENAI_API_KEY
```

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Uses OpenAI model
message = client.messages.create(
    model="gpt-4",  # Resolves to "openai/gpt-4"
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Flow: Anthropic format → Translate → OpenAI API → Translate back
```

### Example 3: Mixed Configuration

```yaml
# config.yaml
model_list:
  # Pass-through to Anthropic
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  # Translate to OpenAI
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
  
  # Translate to Azure
  - model_name: azure-gpt
    litellm_params:
      model: azure/gpt-4
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
```

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# Pass-through
message1 = client.messages.create(
    model="claude",  # → Anthropic API (no translation)
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Translation
message2 = client.messages.create(
    model="gpt-4",  # → OpenAI API (with translation)
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)

# Translation
message3 = client.messages.create(
    model="azure-gpt",  # → Azure API (with translation)
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

## How Provider is Determined

### From Model Name

```python
# LiteLLM extracts provider from model name

"anthropic/claude-3-5-sonnet" → provider = "anthropic"
"openai/gpt-4"                → provider = "openai"
"azure/gpt-4"                 → provider = "azure"
"bedrock/claude-v2"           → provider = "bedrock"
"gpt-4"                       → provider = "openai" (default)
```

### Provider Config Lookup

```python
# litellm/utils.py

def get_provider_anthropic_messages_config(model, provider):
    """
    Check if provider has Anthropic-specific message handling
    """
    if provider == "anthropic":
        # Return Anthropic config (pass-through)
        return AnthropicMessagesConfig()
    elif provider == "vertex_ai" and "claude" in model:
        # Vertex AI with Claude also uses Anthropic format
        return VertexAnthropicMessagesConfig()
    else:
        # No Anthropic-specific config
        # → Use adapter for translation
        return None
```

## Decision Logic Summary

```python
def decide_routing(model_name):
    """
    Pseudo-code for routing decision
    """
    # Step 1: Determine provider from model name
    provider = extract_provider(model_name)
    # "anthropic/claude-3-5-sonnet" → "anthropic"
    # "gpt-4" → "openai"
    
    # Step 2: Check if provider has Anthropic message support
    has_anthropic_support = provider in ["anthropic", "vertex_ai_anthropic"]
    
    # Step 3: Route accordingly
    if has_anthropic_support:
        # Pass-through (no translation)
        return "PASS_THROUGH"
    else:
        # Translation required
        return "TRANSLATE"
```

## Visual Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│ Request arrives at /v1/messages                             │
│ Model: ???                                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Extract model name   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Determine provider   │
              └──────────┬───────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Provider =   │          │ Provider =   │
    │ "anthropic"  │          │ "openai"     │
    │ "vertex_ai"  │          │ "azure"      │
    │ (with Claude)│          │ "bedrock"    │
    └──────┬───────┘          │ etc.         │
           │                  └──────┬───────┘
           │                         │
           ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ PASS-THROUGH │          │  TRANSLATE   │
    │              │          │              │
    │ No format    │          │ Anthropic →  │
    │ conversion   │          │ OpenAI       │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Anthropic    │          │ OpenAI/Azure │
    │ API          │          │ API          │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           ▼                         ▼
    ┌──────────────┐          ┌──────────────┐
    │ Return as-is │          │ Translate    │
    │              │          │ OpenAI →     │
    │              │          │ Anthropic    │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           └────────────┬────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │ Return to client     │
              │ (Anthropic format)   │
              └──────────────────────┘
```

## Key Files

### Decision Logic
- **`litellm/llms/anthropic/experimental_pass_through/messages/handler.py`**
  - Line ~176: Checks if `anthropic_messages_provider_config` exists
  - If None → Translation
  - If exists → Pass-through

### Pass-Through Implementation
- **`litellm/llms/anthropic/experimental_pass_through/messages/transformation.py`**
  - No transformation, just validation

### Translation Implementation
- **`litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`**
  - `translate_anthropic_to_openai()` - Request translation
  - `translate_openai_to_anthropic()` - Response translation

### Provider Detection
- **`litellm/utils.py`**
  - `get_provider_anthropic_messages_config()` - Returns config or None

## Testing the Routing

### Test Pass-Through

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# This should pass through
message = client.messages.create(
    model="anthropic/claude-3-5-sonnet-20241022",
    max_tokens=10,
    messages=[{"role": "user", "content": "Say 'pass-through'"}]
)

print(message.content[0].text)
# Should contain "pass-through" or similar
```

### Test Translation

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# This should translate
message = client.messages.create(
    model="gpt-4",
    max_tokens=10,
    messages=[{"role": "user", "content": "Say 'translated'"}]
)

print(message.content[0].text)
# Should contain "translated" or similar
# Response will be in Anthropic format even though it came from OpenAI
```

## Summary

**The decision is based on the model name:**

1. **Model starts with `anthropic/`** → Pass-through (no translation)
2. **Model is anything else** → Translation (Anthropic ↔ OpenAI)

**Key decision point in code:**
```python
# litellm/llms/anthropic/experimental_pass_through/messages/handler.py (line ~182)

if anthropic_messages_provider_config is None:
    # Translation path
    return adapter.translate(...)
else:
    # Pass-through path
    return pass_through(...)
```

**This allows you to:**
- Use Anthropic SDK with any provider
- Keep your code format unchanged
- Switch providers by changing model name
- Mix and match providers in same application
