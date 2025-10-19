# LiteLLM + Anthropic: Complete Summary

## What is LiteLLM?

LiteLLM is a unified interface library and proxy server that allows you to call 100+ LLM providers (OpenAI, Anthropic, Azure, Bedrock, etc.) using a consistent format. It provides **bidirectional translation** between OpenAI and Anthropic API formats.

## Two Ways to Use LiteLLM with Anthropic

### 1. Forward Translation: OpenAI Format → Anthropic API

**Your code uses OpenAI format, LiteLLM translates to Anthropic**

```python
from litellm import completion

# OpenAI-compatible code
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

**Use when:**
- Building new applications
- Want unified interface across providers
- Need easy provider switching

### 2. Reverse Translation: Anthropic Format → Any Provider

**Your code uses Anthropic format, LiteLLM routes to any provider**

```python
import anthropic

# Point Anthropic SDK to LiteLLM proxy
client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000"
)

# Native Anthropic format, routes to OpenAI!
message = client.messages.create(
    model="gpt-4",  # Can be any provider
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

**Use when:**
- Have existing Anthropic SDK code
- Want to migrate providers without code changes
- Need proxy features with Anthropic format

## How It Works

### Forward Translation Flow

```
Your Code (OpenAI format)
    ↓
LiteLLM Library
    ↓ Translates: OpenAI → Anthropic
Anthropic API
    ↓ Anthropic response
LiteLLM Library
    ↓ Translates: Anthropic → OpenAI
Your Code (OpenAI format response)
```

### Reverse Translation Flow

```
Your Code (Anthropic SDK)
    ↓ Anthropic format
LiteLLM Proxy (/v1/messages)
    ↓ Translates: Anthropic → OpenAI
OpenAI API (or any provider)
    ↓ OpenAI response
LiteLLM Proxy
    ↓ Translates: OpenAI → Anthropic
Your Code (Anthropic format response)
```

## Key Features

### Supported Capabilities

| Feature | Forward | Reverse | Notes |
|---------|---------|---------|-------|
| Basic Completion | ✅ | ✅ | Full support |
| Streaming | ✅ | ✅ | Real-time responses |
| Async | ✅ | ✅ | Non-blocking calls |
| System Prompts | ✅ | ✅ | Translated automatically |
| Tool/Function Calling | ✅ | ✅ | Format conversion |
| Vision (Images) | ✅ | ✅ | Multi-modal support |
| Prompt Caching | ✅ | ✅ | Cost optimization |
| Extended Thinking | ✅ | ✅ | Claude Sonnet 4 |
| Load Balancing | ✅ | ✅ | Via Router/Proxy |
| Fallbacks | ✅ | ✅ | Automatic retry |
| Cost Tracking | ✅ | ✅ | Built-in |
| Rate Limiting | ✅ | ✅ | Proxy feature |

### Available Models

```python
# Claude 3.5 Sonnet - Best balance of intelligence and speed
"anthropic/claude-3-5-sonnet-20241022"

# Claude 3.5 Haiku - Fastest and most affordable
"anthropic/claude-3-5-haiku-20241022"

# Claude 3 Opus - Most capable for complex tasks
"anthropic/claude-3-opus-20240229"

# Claude Sonnet 4 - Extended thinking for reasoning
"anthropic/claude-3-7-sonnet-20250219"
```

## Quick Start Guide

### Option 1: Library (Forward Translation)

```bash
# Install
pip install litellm

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Use in code
python anthropic_quickstart.py
```

### Option 2: Proxy (Reverse Translation)

```bash
# Install
pip install 'litellm[proxy]'

# Start proxy (no config needed!)
litellm --model anthropic/claude-3-5-sonnet-20241022

# Endpoint available at:
# http://localhost:4000/v1/messages
```

## Proxy Endpoints

### Available by Default (No Config Required)

When you start the proxy with:
```bash
litellm --model anthropic/claude-3-5-sonnet-20241022
```

These endpoints are immediately available:

1. **`/v1/messages`** - Anthropic-compatible endpoint
   - URL: `http://localhost:4000/v1/messages`
   - Format: Native Anthropic
   - Routes to: Any configured provider

2. **`/v1/chat/completions`** - OpenAI-compatible endpoint
   - URL: `http://localhost:4000/v1/chat/completions`
   - Format: OpenAI
   - Routes to: Any configured provider

3. **`/anthropic/v1/messages`** - Alternative Anthropic endpoint
   - URL: `http://localhost:4000/anthropic/v1/messages`
   - Format: Native Anthropic
   - Recommended for clarity

### No Configuration File Required

The `/v1/messages` endpoint is registered automatically in the proxy server code:

```python
# litellm/proxy/proxy_server.py (line ~9561)
from litellm.proxy.anthropic_endpoints.endpoints import router as anthropic_router
app.include_router(anthropic_router)  # Registers /v1/messages
```

## Architecture

### Code Structure

```
litellm/
├── main.py                           # Entry point for library
├── router.py                         # Load balancing & routing
├── llms/
│   └── anthropic/
│       ├── chat/
│       │   ├── handler.py           # HTTP calls to Anthropic
│       │   └── transformation.py    # OpenAI ↔ Anthropic translation
│       ├── common_utils.py          # Utilities
│       └── experimental_pass_through/
│           ├── messages/            # Anthropic → Anthropic (pass-through)
│           └── adapters/            # Anthropic → OpenAI (translation)
└── proxy/
    ├── proxy_server.py              # FastAPI app
    ├── anthropic_endpoints/         # /v1/messages endpoint
    └── auth/                        # Authentication
```

### Translation Components

**Forward Translation (OpenAI → Anthropic):**
- `litellm/llms/anthropic/chat/transformation.py`
  - Converts OpenAI messages to Anthropic format
  - Maps parameters (stop → stop_sequences, etc.)
  - Handles tools, system prompts, vision

**Reverse Translation (Anthropic → OpenAI):**
- `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
  - Converts Anthropic messages to OpenAI format
  - Maps parameters back
  - Handles response conversion

## Example Use Cases

### Use Case 1: New Application (Forward)

**Scenario:** Building a new chatbot, want flexibility to switch providers

**Solution:** Use LiteLLM library with OpenAI format

```python
from litellm import completion

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Easy to switch: model="openai/gpt-4"
```

### Use Case 2: Existing Anthropic Code (Reverse)

**Scenario:** Have production code using Anthropic SDK, want to test OpenAI

**Solution:** Point Anthropic SDK to LiteLLM proxy

```python
import anthropic

# ONLY CHANGE: 2 lines
client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000"  # Point to proxy
)

# SAME CODE - now routes to OpenAI!
message = client.messages.create(
    model="gpt-4",  # Changed model name
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Use Case 3: Production with Fallbacks

**Scenario:** Need high availability with automatic fallbacks

**Solution:** Use LiteLLM proxy with config

```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-fallback
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

fallbacks:
  - claude: ["claude-fallback"]

router_settings:
  num_retries: 3
```

### Use Case 4: Cost Tracking & Rate Limiting

**Scenario:** Need to track costs per team and enforce rate limits

**Solution:** Use proxy with database

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
  max_budget: 1000  # $1000 max
  budget_duration: 30d
```

## Parameter Mapping

### OpenAI → Anthropic (Forward)

| OpenAI Parameter | Anthropic Parameter | Transformation |
|-----------------|---------------------|----------------|
| `messages` | `messages` + `system` | System messages extracted |
| `max_tokens` | `max_tokens` | Direct (required by Anthropic) |
| `temperature` | `temperature` | Direct |
| `top_p` | `top_p` | Direct |
| `stop` | `stop_sequences` | Renamed |
| `tools` | `tools` | Format converted |
| `tool_choice` | `tool_choice` | Format converted |
| `stream` | `stream` | Direct |
| `user` | `metadata.user_id` | Moved to metadata |

### Anthropic → OpenAI (Reverse)

| Anthropic Parameter | OpenAI Parameter | Transformation |
|--------------------|------------------|----------------|
| `system` | `messages[0]` | Converted to system message |
| `max_tokens` | `max_tokens` | Direct |
| `temperature` | `temperature` | Direct |
| `top_p` | `top_p` | Direct |
| `stop_sequences` | `stop` | Renamed |
| `tools` | `tools` | Format converted |
| `tool_choice` | `tool_choice` | Format converted |
| `stream` | `stream` | Direct |
| `metadata.user_id` | `user` | Extracted |

## Benefits of Using LiteLLM

### For New Projects (Forward Translation)

✅ **Unified Interface**: Same code works with 100+ providers
✅ **Easy Switching**: Change one parameter to switch providers
✅ **Built-in Features**: Retry, fallback, load balancing
✅ **Cost Tracking**: Automatic token and cost tracking
✅ **Production Ready**: Used by thousands of companies

### For Existing Anthropic Code (Reverse Translation)

✅ **No Code Changes**: Keep using Anthropic SDK (change 2 lines)
✅ **Provider Flexibility**: Route to OpenAI, Azure, etc.
✅ **Proxy Features**: Add auth, rate limiting, cost tracking
✅ **Easy Migration**: Test new providers without rewrites
✅ **A/B Testing**: Compare providers with same code

### For Production Deployments

✅ **High Availability**: Automatic fallbacks and retries
✅ **Load Balancing**: Distribute across multiple keys/deployments
✅ **Cost Control**: Budgets, rate limits per team/user
✅ **Observability**: Logging, metrics, tracing
✅ **Security**: Authentication, key management
✅ **Caching**: Redis/in-memory caching for cost savings

## Files in This Directory

### Quick Start
- **[anthropic_quickstart.py](./anthropic_quickstart.py)** - 10-line minimal example

### Comprehensive Examples
- **[anthropic_example.py](./anthropic_example.py)** - 13 detailed examples
- **[complete_anthropic_proxy_example.py](./complete_anthropic_proxy_example.py)** - Real-world examples

### Migration & Proxy
- **[anthropic_to_proxy_migration.py](./anthropic_to_proxy_migration.py)** - Before/after comparison
- **[reverse_translation_example.py](./reverse_translation_example.py)** - Anthropic format → Any provider
- **[proxy_config.yaml](./proxy_config.yaml)** - Configuration examples

### Documentation
- **[README.md](./README.md)** - Overview and quick start
- **[ANTHROPIC_GUIDE.md](./ANTHROPIC_GUIDE.md)** - Complete integration guide
- **[anthropic_code_flow.md](./anthropic_code_flow.md)** - Internal code flow
- **[ENDPOINT_AVAILABILITY.md](./ENDPOINT_AVAILABILITY.md)** - Proxy endpoint details
- **[SUMMARY.md](./SUMMARY.md)** - This file

## Common Questions

### Q: Do I need a config file to use the proxy?
**A:** No! The `/v1/messages` endpoint is available by default. Just run:
```bash
litellm --model anthropic/claude-3-5-sonnet-20241022
```

### Q: Can I use Anthropic SDK with OpenAI models?
**A:** Yes! Point the Anthropic SDK to the proxy:
```python
client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000"
)
message = client.messages.create(model="gpt-4", ...)  # Routes to OpenAI!
```

### Q: Which approach should I use?
**A:** 
- **New projects**: Use forward translation (OpenAI format)
- **Existing Anthropic code**: Use reverse translation (Anthropic format via proxy)
- **Production**: Use proxy for both (adds features)

### Q: Does this work with streaming?
**A:** Yes! Both forward and reverse translation support streaming.

### Q: Can I use both formats at the same time?
**A:** Yes! The proxy supports both `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic) simultaneously.

### Q: What about tool calling?
**A:** Fully supported in both directions. LiteLLM automatically converts tool formats.

### Q: Is there any latency overhead?
**A:** Minimal. Translation is fast. Proxy adds ~10-50ms depending on features enabled.

### Q: Can I use this in production?
**A:** Yes! LiteLLM is used by thousands of companies in production.

## Getting Started

### 1. Choose Your Approach

**Forward Translation (OpenAI format):**
```bash
pip install litellm
python anthropic_quickstart.py
```

**Reverse Translation (Anthropic format):**
```bash
pip install 'litellm[proxy]'
litellm --model anthropic/claude-3-5-sonnet-20241022
# Use Anthropic SDK with base_url="http://localhost:4000"
```

### 2. Explore Examples

- Start with `anthropic_quickstart.py`
- Try examples in `anthropic_example.py`
- Read `ANTHROPIC_GUIDE.md` for details
- Check `reverse_translation_example.py` for proxy usage

### 3. Deploy to Production

- Create `config.yaml` with your models
- Add authentication and rate limiting
- Set up database for cost tracking
- Configure fallbacks and load balancing

## Resources

- **LiteLLM Docs**: https://docs.litellm.ai/
- **Anthropic Docs**: https://docs.anthropic.com/
- **GitHub**: https://github.com/BerriAI/litellm
- **Discord**: https://discord.gg/wuPM9dRgDw

## Summary

LiteLLM provides **bidirectional translation** between OpenAI and Anthropic formats:

1. **Forward**: OpenAI format → Anthropic API (for new projects)
2. **Reverse**: Anthropic format → Any provider (for existing code)

Both approaches give you:
- Multi-provider support
- Production features (fallbacks, tracking, auth)
- Easy migration and testing
- Minimal code changes

The `/v1/messages` endpoint is **available by default** - no config required!

**Ready to start?** Run `python anthropic_quickstart.py` or explore the examples!
