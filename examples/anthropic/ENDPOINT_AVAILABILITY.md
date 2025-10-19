# Anthropic Endpoint Availability in LiteLLM Proxy

## Quick Answer

**YES**, the `/v1/messages` endpoint is available **by default** when you start the LiteLLM proxy server. No configuration file is required.

## Available Endpoints

When you start the proxy with:
```bash
litellm --model anthropic/claude-3-5-sonnet-20241022
```

The following Anthropic-compatible endpoints are automatically available:

### 1. `/v1/messages` (Default)
```bash
http://localhost:4000/v1/messages
```

This is the **standard Anthropic endpoint** that accepts native Anthropic format.

### 2. `/anthropic/v1/messages` (Recommended)
```bash
http://localhost:4000/anthropic/v1/messages
```

This is the **recommended endpoint** with the `/anthropic` prefix for clarity.

## How It Works

The endpoint is registered automatically in the proxy server:

```python
# In litellm/proxy/proxy_server.py (line ~9561)
from litellm.proxy.anthropic_endpoints.endpoints import router as anthropic_router

app.include_router(anthropic_router)  # Registers /v1/messages
```

The route is defined in `litellm/proxy/anthropic_endpoints/endpoints.py`:

```python
@router.post(
    "/v1/messages",
    tags=["[beta] Anthropic `/v1/messages`"],
    dependencies=[Depends(user_api_key_auth)],
)
async def anthropic_response(...):
    """
    Anthropic-compatible endpoint that accepts native Anthropic format
    """
```

## Usage Examples

### Without Config File (Simplest)

```bash
# Start proxy with just a model
litellm --model anthropic/claude-3-5-sonnet-20241022

# The endpoint is immediately available at:
# http://localhost:4000/v1/messages
```

### With Config File (Recommended for Production)

```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

```bash
litellm --config config.yaml

# Endpoints available:
# http://localhost:4000/v1/messages
# http://localhost:4000/anthropic/v1/messages
```

## Using the Endpoint

### With Anthropic SDK

```python
import anthropic

# Point to LiteLLM proxy
client = anthropic.Anthropic(
    api_key="sk-1234",  # Your proxy key (or "anything" for testing)
    base_url="http://localhost:4000"  # Proxy will use /v1/messages
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### With cURL

```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Endpoint Behavior

### What the Endpoint Does

1. **Accepts**: Native Anthropic `/v1/messages` format
2. **Routes**: To configured model (can be Anthropic, OpenAI, Azure, etc.)
3. **Translates**: If routing to non-Anthropic provider, translates format
4. **Returns**: Response in Anthropic format

### Example Flow

```
Your Request (Anthropic format)
    ↓
http://localhost:4000/v1/messages
    ↓
LiteLLM Proxy
    ├─ If model is Anthropic → Pass through
    └─ If model is OpenAI → Translate format
    ↓
Provider API
    ↓
LiteLLM Proxy (translates response back)
    ↓
Your Response (Anthropic format)
```

## Configuration Options

### No Config (Default Behavior)

```bash
# Start with CLI args
litellm --model anthropic/claude-3-5-sonnet-20241022

# Endpoint available immediately
# Uses ANTHROPIC_API_KEY from environment
```

### With Config (More Control)

```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  # Route Anthropic format to OpenAI
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

# Optional: Enable authentication
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

### With Database (Virtual Keys)

```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://user:pass@localhost/litellm
```

Then generate virtual keys:
```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-key" \
  -H "Content-Type: application/json" \
  -d '{"models": ["claude"]}'
```

## Authentication

### Without Config (Open Access)
```bash
litellm --model anthropic/claude-3-5-sonnet-20241022

# No authentication required
# Use any API key or "anything"
```

### With Master Key (Recommended)
```bash
export LITELLM_MASTER_KEY="sk-1234"
litellm --model anthropic/claude-3-5-sonnet-20241022

# Requires valid key
curl http://localhost:4000/v1/messages \
  -H "x-api-key: sk-1234" \
  ...
```

### With Virtual Keys (Production)
```yaml
# config.yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
```

Generate keys with budgets, rate limits, etc.

## Checking Available Endpoints

### View All Endpoints

Visit the Swagger docs:
```
http://localhost:4000/docs
```

Look for:
- `[beta] Anthropic /v1/messages` section
- Shows the `/v1/messages` endpoint

### Test Endpoint Availability

```bash
# Health check
curl http://localhost:4000/health

# Test Anthropic endpoint
curl http://localhost:4000/v1/messages \
  -H "x-api-key: anything" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

## Summary

| Question | Answer |
|----------|--------|
| Is `/v1/messages` available by default? | ✅ YES |
| Do I need a config file? | ❌ NO (but recommended for production) |
| What's the endpoint URL? | `http://localhost:4000/v1/messages` |
| Alternative endpoint? | `http://localhost:4000/anthropic/v1/messages` |
| Authentication required? | Only if you set `master_key` |
| Can route to non-Anthropic providers? | ✅ YES (OpenAI, Azure, etc.) |

## Common Scenarios

### Scenario 1: Quick Test (No Config)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --model anthropic/claude-3-5-sonnet-20241022

# Endpoint ready at http://localhost:4000/v1/messages
```

### Scenario 2: Production (With Config)
```bash
# config.yaml with multiple models, auth, etc.
litellm --config config.yaml

# Endpoint ready at http://localhost:4000/v1/messages
```

### Scenario 3: Route to OpenAI (With Config)
```yaml
# config.yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

```python
# Use Anthropic SDK, routes to OpenAI!
client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="gpt-4",  # Routes to OpenAI
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Troubleshooting

### Endpoint Not Found (404)

**Problem**: Getting 404 on `/v1/messages`

**Solutions**:
1. Check proxy is running: `curl http://localhost:4000/health`
2. Check correct port: Default is 4000
3. Try alternative endpoint: `/anthropic/v1/messages`

### Authentication Error (401)

**Problem**: Getting 401 Unauthorized

**Solutions**:
1. If using `master_key`, include it: `-H "x-api-key: sk-master-key"`
2. If no auth, use any key: `-H "x-api-key: anything"`
3. Check `general_settings.master_key` in config

### Model Not Found

**Problem**: Model not available

**Solutions**:
1. Check model is in config: `model_list`
2. Check API key is set: `ANTHROPIC_API_KEY`
3. View available models: `curl http://localhost:4000/models`

## References

- Proxy Server Code: `litellm/proxy/proxy_server.py` (line ~9561)
- Endpoint Definition: `litellm/proxy/anthropic_endpoints/endpoints.py`
- Documentation: https://docs.litellm.ai/docs/anthropic_completion
