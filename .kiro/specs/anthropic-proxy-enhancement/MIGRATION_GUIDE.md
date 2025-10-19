# Anthropic Proxy Enhancement Migration Guide

## Overview

The LiteLLM proxy's `/v1/messages` endpoint has been enhanced with a new transformation mode that enables routing Anthropic-formatted requests to any OpenAI-compatible provider. This guide helps you understand the changes and migrate your existing setup if needed.

## What's New

### Transformation Mode (Default)

The new default behavior transforms Anthropic Messages API requests to OpenAI format, allowing you to:

- Route Claude requests to OpenAI, Azure OpenAI, or any other provider
- Use LiteLLM's full routing capabilities (load balancing, fallbacks, rate limiting)
- Maintain Anthropic API compatibility for clients (Claude Code, Anthropic SDK)
- Support all Anthropic features: text/image content, tool calling, streaming

**Example Configuration:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: gpt-4  # Route to OpenAI
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-opus-20240229
    litellm_params:
      model: azure/gpt-4  # Route to Azure
      api_base: https://your-azure-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY

# Transformation mode is enabled by default
general_settings:
  anthropic_transformation_enabled: true  # Optional, true by default
```

### Pass-Through Mode (Backward Compatible)

The original pass-through behavior is still available for users who need it. This mode routes requests directly to the Anthropic API without transformation.

**Example Configuration:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

# Disable transformation to use pass-through mode
general_settings:
  anthropic_transformation_enabled: false
```

## Migration Scenarios

### Scenario 1: You're Using Anthropic API Directly (Pass-Through)

**Current Setup:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Migration Options:**

#### Option A: Keep Pass-Through Mode (No Changes Required)
Add this to your configuration to maintain existing behavior:
```yaml
general_settings:
  anthropic_transformation_enabled: false
```

#### Option B: Enable Transformation Mode (Recommended)
No configuration changes needed! The new transformation mode will:
- Still route to Anthropic API
- Add support for routing to other providers
- Enable all LiteLLM routing features

Your existing configuration will work as-is with enhanced capabilities.

### Scenario 2: You Want to Route Claude Requests to OpenAI

**New Setup:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

# Transformation mode is enabled by default
general_settings:
  anthropic_transformation_enabled: true  # Optional
```

**Client Code (No Changes):**
```python
from anthropic import Anthropic

client = Anthropic(
    api_key="your-litellm-key",
    base_url="http://localhost:4000"  # Your LiteLLM proxy
)

# This request will be routed to OpenAI GPT-4
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Scenario 3: You Want Load Balancing Across Providers

**New Setup:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: azure/gpt-4
      api_base: https://your-azure-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
  
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  routing_strategy: simple-shuffle  # Load balance across all three

general_settings:
  anthropic_transformation_enabled: true
```

**Client Code (No Changes):**
```python
# Requests will be load balanced across OpenAI, Azure, and Anthropic
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Scenario 4: You Want Fallbacks

**New Setup:**
```yaml
model_list:
  - model_name: claude-3-sonnet-20240229
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  fallbacks:
    - model: azure/gpt-4
      api_base: https://your-azure-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY

general_settings:
  anthropic_transformation_enabled: true
```

## Breaking Changes

**None!** This enhancement is fully backward compatible:

1. **Existing `/v1/messages` behavior preserved**: Set `anthropic_transformation_enabled: false` to use pass-through mode
2. **Existing OpenAI endpoints unaffected**: `/v1/chat/completions` and other endpoints work exactly as before
3. **Default behavior is opt-in**: New deployments get transformation mode by default, but existing deployments can opt-in gradually
4. **Both modes can coexist**: You can use OpenAI format on `/v1/chat/completions` and Anthropic format on `/v1/messages` simultaneously

## Testing Your Migration

### 1. Test Pass-Through Mode (Backward Compatibility)

```bash
# Set pass-through mode
export ANTHROPIC_TRANSFORMATION_ENABLED=false

# Start proxy
litellm --config config.yaml

# Test with Anthropic SDK
curl -X POST http://localhost:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 2. Test Transformation Mode

```bash
# Enable transformation mode (default)
# No environment variable needed

# Start proxy
litellm --config config.yaml

# Test routing to OpenAI
curl -X POST http://localhost:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 3. Test OpenAI Endpoints (Should Be Unaffected)

```bash
# Test that OpenAI endpoints still work
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration Reference

### General Settings

```yaml
general_settings:
  # Enable/disable Anthropic transformation mode
  # Default: true
  # When true: Transforms Anthropic requests to OpenAI format, routes to any provider
  # When false: Uses pass-through mode, routes directly to Anthropic API
  anthropic_transformation_enabled: true
```

### Environment Variables

```bash
# Set via environment variable (optional)
ANTHROPIC_TRANSFORMATION_ENABLED=true  # or false
```

## Troubleshooting

### Issue: Requests failing after upgrade

**Solution:** Enable pass-through mode to maintain existing behavior:
```yaml
general_settings:
  anthropic_transformation_enabled: false
```

### Issue: Tool calling not working with non-Anthropic providers

**Cause:** Some providers may not support all OpenAI tool calling features.

**Solution:** 
1. Test with OpenAI or Azure OpenAI first
2. Check provider documentation for tool calling support
3. Use pass-through mode with Anthropic API if needed

### Issue: Streaming responses have different format

**Cause:** Transformation mode converts OpenAI streaming chunks to Anthropic SSE events.

**Solution:** This is expected behavior. The Anthropic SDK handles this automatically. If you're using raw HTTP, ensure you're parsing SSE events correctly.

### Issue: Response format doesn't match Anthropic API

**Cause:** Transformation mode converts OpenAI responses to Anthropic format.

**Solution:** This should be transparent to Anthropic SDK clients. If you're seeing issues:
1. Check that you're using the latest Anthropic SDK
2. Verify your client is parsing Anthropic response format
3. Enable debug logging to see the transformation

## Support

For issues or questions:
1. Check the [LiteLLM documentation](https://docs.litellm.ai/)
2. Review the [Anthropic endpoint documentation](https://docs.litellm.ai/docs/proxy/anthropic_completion)
3. Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)

## Summary

- **Default behavior**: Transformation mode enabled (routes to any provider)
- **Backward compatibility**: Set `anthropic_transformation_enabled: false` for pass-through mode
- **No breaking changes**: Existing functionality preserved
- **Gradual migration**: Test transformation mode before fully switching
- **Enhanced capabilities**: Load balancing, fallbacks, multi-provider support
