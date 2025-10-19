# LiteLLM + Anthropic Examples

This directory contains comprehensive examples and documentation for using LiteLLM with Anthropic's Claude models.

## Files Overview

### Quick Start
- **`anthropic_quickstart.py`** - Minimal 10-line example to get started immediately

### Comprehensive Examples
- **`anthropic_example.py`** - 13 detailed examples covering all major features:
  - Basic completion
  - Streaming
  - Async operations
  - Multi-turn conversations
  - System prompts
  - Tool/function calling
  - Vision (image analysis)
  - Advanced parameters
  - Prompt caching
  - Extended thinking (Claude Sonnet 4)
  - Error handling
  - Model comparison
  - Load balancing with Router

### Documentation
- **`ANTHROPIC_GUIDE.md`** - Complete integration guide covering:
  - Architecture overview
  - Supported models
  - Feature details
  - Parameter mapping
  - Error handling
  - Cost tracking
  - Best practices
  - Troubleshooting

- **`anthropic_code_flow.md`** - Deep dive into how LiteLLM works:
  - Step-by-step request flow
  - Code transformation process
  - Response normalization
  - Streaming implementation
  - File references

## Quick Start

### 1. Install LiteLLM
```bash
pip install litellm
```

### 2. Set API Key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run Quick Start
```bash
python anthropic_quickstart.py
```

### 4. Explore Full Examples
```bash
python anthropic_example.py
```

## Key Concepts

### Why Use LiteLLM with Anthropic?

1. **Unified Interface**: Use the same code for OpenAI, Anthropic, Azure, and 100+ providers
2. **Easy Migration**: Switch between providers by changing one parameter
3. **Advanced Features**: Built-in retry logic, load balancing, cost tracking
4. **Production Ready**: Used by thousands of companies in production

### Basic Usage Pattern

```python
from litellm import completion

# Just change the model prefix to switch providers
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",  # Anthropic
    # model="openai/gpt-4",                        # OpenAI
    # model="azure/gpt-4",                         # Azure
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Application                      │
│                    (OpenAI-format code)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      LiteLLM Library                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  main.py - Entry Point & Provider Detection         │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  anthropic/chat/transformation.py                    │   │
│  │  • Convert OpenAI → Anthropic format                 │   │
│  │  • Handle system messages, tools, parameters         │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  anthropic/chat/handler.py                           │   │
│  │  • Make HTTP requests                                │   │
│  │  • Handle streaming                                  │   │
│  │  • Error handling                                    │   │
│  └────────────────────┬─────────────────────────────────┘   │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Anthropic API                              │
│              https://api.anthropic.com                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Response Transformation                     │
│              Anthropic → OpenAI format                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│              (Receives OpenAI-format response)               │
└─────────────────────────────────────────────────────────────┘
```

## Supported Features

| Feature | Supported | Example |
|---------|-----------|---------|
| Basic Completion | ✅ | `anthropic_example.py` - Example 1 |
| Streaming | ✅ | `anthropic_example.py` - Example 2 |
| Async | ✅ | `anthropic_example.py` - Example 3 |
| Multi-turn Chat | ✅ | `anthropic_example.py` - Example 4 |
| System Prompts | ✅ | `anthropic_example.py` - Example 5 |
| Function Calling | ✅ | `anthropic_example.py` - Example 6 |
| Vision | ✅ | `anthropic_example.py` - Example 7 |
| Prompt Caching | ✅ | `anthropic_example.py` - Example 9 |
| Extended Thinking | ✅ | `anthropic_example.py` - Example 10 |
| PDF Analysis | ✅ | See `ANTHROPIC_GUIDE.md` |
| Computer Use | ✅ | See `ANTHROPIC_GUIDE.md` |
| Load Balancing | ✅ | `anthropic_example.py` - Example 13 |

## Available Models

```python
# Claude 3.5 Sonnet - Best balance of intelligence and speed
"anthropic/claude-3-5-sonnet-20241022"

# Claude 3.5 Haiku - Fastest and most affordable
"anthropic/claude-3-5-haiku-20241022"

# Claude 3 Opus - Most capable for complex tasks
"anthropic/claude-3-opus-20240229"

# Claude Sonnet 4 - Extended thinking for reasoning tasks
"anthropic/claude-3-7-sonnet-20250219"
```

## Common Use Cases

### 1. Simple Chat
```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 2. With System Prompt
```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ]
)
```

### 3. Streaming
```python
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### 4. Function Calling
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {...}
    }
}]

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[...],
    tools=tools
)
```

## Next Steps

1. **Start Simple**: Run `anthropic_quickstart.py`
2. **Explore Features**: Try examples in `anthropic_example.py`
3. **Read the Guide**: Check `ANTHROPIC_GUIDE.md` for details
4. **Understand Internals**: Read `anthropic_code_flow.md`
5. **Build Your App**: Use LiteLLM in your project!

## Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)

## Need Help?

- Check the examples in this directory
- Read the troubleshooting section in `ANTHROPIC_GUIDE.md`
- Join the [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)
- Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)
