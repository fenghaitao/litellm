# Enhanced Anthropic API Endpoints

This module provides comprehensive Anthropic API compatibility for the LiteLLM proxy, enabling full support for Claude Code and other Anthropic clients.

## Features

### 🚀 Enhanced Endpoints
- **`/v1/messages`** - Complete Anthropic Messages API with validation and model mapping
- **`/v1/models`** - List available Anthropic models
- **`/v1/count_tokens`** - Enhanced token counting with validation
- **`/v1/count_request_tokens`** - Detailed token breakdown
- **`/v1/check_context_limit`** - Context limit validation

### 🔍 Advanced Validation
- **Message Structure** - Comprehensive validation of message format and content blocks
- **Tool Definitions** - Full validation of tool schemas and parameters
- **Content Blocks** - Support for text, tool_use, tool_result, and image blocks
- **System Messages** - Validation of system prompts and instructions

### 🎯 Model Tier Mapping
- **Anthropic Models → Tiers** - Map models to big/medium/small tiers
- **Provider Resolution** - Automatically resolve to provider-specific models
- **Dynamic Configuration** - Support for multiple provider backends

### 📡 Enhanced Streaming
- **Proper SSE Format** - Full Server-Sent Events implementation
- **Event Types** - Complete support for all Anthropic streaming events
- **Content Blocks** - Proper handling of text and tool streaming

## Quick Start

### 1. Basic Usage
The enhanced endpoints work automatically with existing LiteLLM proxy setups:

```bash
# Start LiteLLM proxy
litellm --config config.yaml

# Use Anthropic API format
curl -X POST "http://localhost:4000/v1/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [{"role": "user", "content": "Hello Claude!"}],
    "max_tokens": 1000
  }'
```

### 2. Model Mapping Configuration

Add Anthropic configuration to your LiteLLM config:

```yaml
# config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  anthropic:
    model_mappings:
      # Define which tier each Anthropic model belongs to
      anthropic_models:
        claude-3-opus-20240229: big
        claude-3-sonnet-20240229: medium
        claude-3-haiku-20240307: small
        claude-3-5-sonnet-20241022: medium
        
      # Define provider-specific model mappings
      providers:
        openai:
          prefix: "openai/"
          default: true
          models:
            big: "gpt-4o"
            medium: "gpt-4o-mini"
            small: "gpt-3.5-turbo"
            
        github_copilot:
          prefix: "github_copilot/"
          models:
            big: "gpt-4o"
            medium: "gpt-4o"
            small: "gpt-4o-mini"
          extra_headers:
            Editor-Version: "vscode/1.85.0"
            Editor-Plugin-Version: "copilot-chat/0.11.1"
            User-Agent: "GitHubCopilot/1.0"
            Copilot-Integration-Id: "vscode-chat"
```

### 3. Environment Variables

Configure providers using environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic (for direct access)
export ANTHROPIC_API_KEY="your-anthropic-key"

# GitHub Copilot
export GITHUB_TOKEN="your-github-token"

# ModelScope
export MODELSCOPE_API_KEY="your-modelscope-key"

# Custom model mapping (JSON format)
export ANTHROPIC_MODEL_MAPPING='{"openai": {"models": {"big": "gpt-4o"}}}'
```

## Advanced Usage

### Tool Calling
Full support for Anthropic tool calling format:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-litellm-key",
    base_url="http://localhost:4000"
)

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    tools=[{
        "name": "get_weather", 
        "description": "Get weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }],
    messages=[{
        "role": "user", 
        "content": "What's the weather in Paris?"
    }]
)
```

### Streaming
Enhanced streaming with proper SSE events:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-litellm-key",
    base_url="http://localhost:4000"
)

with client.messages.stream(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            print(event.delta.text, end="")
```

### Token Counting
Enhanced token counting with validation:

```bash
curl -X POST "http://localhost:4000/v1/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [{"role": "user", "content": "Hello Claude!"}],
    "system": "You are a helpful assistant."
  }'

# Response: {"input_tokens": 25}
```

### Context Limit Checking
Validate requests against model context limits:

```bash
curl -X POST "http://localhost:4000/v1/check_context_limit" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-key" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "messages": [{"role": "user", "content": "Very long text..."}],
    "max_tokens": 1000
  }'

# Response:
# {
#   "fits_context": true,
#   "input_tokens": 1500,
#   "max_context_tokens": 200000,
#   "remaining_tokens": 198500,
#   "max_output_tokens": 1000
# }
```

## Claude Code Integration

The enhanced Anthropic endpoints provide full compatibility with Claude Code IDE extension:

1. **Set base URL** in Claude Code settings to your LiteLLM proxy
2. **Configure API key** to use your LiteLLM key
3. **Use any Anthropic model name** - automatic mapping to your configured providers

## Supported Providers

The model mapping system supports any LiteLLM-compatible provider:

- **OpenAI** (`openai/`)
- **Anthropic** (`anthropic/`)
- **GitHub Copilot** (`github_copilot/`)
- **ModelScope** (`dashscope/`)
- **Azure OpenAI** (`azure/`)
- **And 100+ more...**

## Error Handling

Enhanced error responses with Anthropic-compatible format:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Tool at index 0 must have a 'name' field"
  }
}
```

## Migration from Basic Anthropic Support

The enhanced endpoints are backward compatible. Existing `/v1/messages` requests will continue to work with additional features:

- ✅ Enhanced validation
- ✅ Model mapping
- ✅ Better streaming
- ✅ Additional endpoints

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude Code   │────│  LiteLLM Proxy  │────│   Provider      │
│   Extension     │    │  Enhanced API   │    │   (OpenAI/etc)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌──────┴──────┐
                       │   Features   │
                       │ • Validation │
                       │ • Mapping    │
                       │ • Streaming  │
                       │ • Tokens     │
                       └─────────────┘
```

## Contributing

The enhanced Anthropic endpoints are modular and extensible:

- **`validation.py`** - Add new validation rules
- **`model_mapping.py`** - Extend model mapping logic
- **`config.py`** - Add configuration options

## Support

For issues and questions:
- Check the [LiteLLM documentation](https://docs.litellm.ai/)
- Report bugs in the [GitHub repository](https://github.com/BerriAI/litellm)
- Join the community discussions