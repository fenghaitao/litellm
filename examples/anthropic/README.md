# LiteLLM + Anthropic Examples

Complete examples and documentation for using LiteLLM with Anthropic's Claude models, including **bidirectional translation** between OpenAI and Anthropic formats.

## 📚 Files Overview

### Quick Start
- **[anthropic_quickstart.py](./anthropic_quickstart.py)** - 10-line minimal example to get started

### Comprehensive Examples
- **[anthropic_example.py](./anthropic_example.py)** - 13 detailed examples covering:
  - Basic completion, streaming, async
  - Multi-turn conversations, system prompts
  - Tool/function calling, vision
  - Prompt caching, extended thinking
  - Error handling, load balancing

### Migration & Proxy
- **[anthropic_to_proxy_migration.py](./anthropic_to_proxy_migration.py)** - Side-by-side comparison:
  - Original Anthropic SDK vs LiteLLM Proxy
  - Migration checklist and examples
  - Benefits comparison

- **[complete_anthropic_proxy_example.py](./complete_anthropic_proxy_example.py)** - Real-world examples:
  - Chatbot implementation
  - Batch processing
  - Tool calling
  - Before/after comparisons

- **[proxy_config.yaml](./proxy_config.yaml)** - Complete proxy configuration examples

### Reverse Translation ⭐ NEW
- **[reverse_translation_example.py](./reverse_translation_example.py)** - Use Anthropic SDK with any provider:
  - Keep Anthropic format, route to OpenAI/Azure
  - Bidirectional translation explained
  - Migration without code changes

### Documentation
- **[ANTHROPIC_GUIDE.md](./ANTHROPIC_GUIDE.md)** - Complete integration guide:
  - Architecture overview
  - Supported models and features
  - Parameter mapping
  - Best practices and troubleshooting

- **[anthropic_code_flow.md](./anthropic_code_flow.md)** - Deep dive into internals:
  - Step-by-step request flow
  - Code transformation process
  - File references

## 🚀 Quick Start

### 1. Install
```bash
pip install litellm
```

### 2. Set API Key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run
```bash
python anthropic_quickstart.py
```

## 🔄 Two Ways to Use LiteLLM with Anthropic

### Option 1: OpenAI Format → Anthropic (Forward Translation)

Use OpenAI-compatible code that works with 100+ providers:

```python
from litellm import completion

# Your code uses OpenAI format
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Switch providers by changing model name
# model="openai/gpt-4"
# model="azure/gpt-4"
```

**Use when:**
- Building new applications
- Want unified interface across providers
- Need easy provider switching

### Option 2: Anthropic Format → Any Provider (Reverse Translation) ⭐

Keep using Anthropic SDK, route to any provider via proxy:

```python
import anthropic

# Point Anthropic SDK to LiteLLM proxy
client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000/anthropic"
)

# Use native Anthropic format
message = client.messages.create(
    model="gpt-4",  # Routes to OpenAI!
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Use when:**
- Have existing Anthropic SDK code
- Want to migrate providers without code changes
- Need proxy features (fallbacks, tracking) with Anthropic format

## 📊 Translation Flows

### Forward Translation (OpenAI → Anthropic)
```
Your Code (OpenAI format)
    ↓
LiteLLM Library
    ↓ Translates to Anthropic format
Anthropic API
    ↓ Anthropic response
LiteLLM Library
    ↓ Translates to OpenAI format
Your Code (OpenAI format)
```

### Reverse Translation (Anthropic → OpenAI)
```
Your Code (Anthropic SDK)
    ↓ Anthropic format
LiteLLM Proxy
    ↓ Translates to OpenAI format
OpenAI API (or any provider)
    ↓ OpenAI response
LiteLLM Proxy
    ↓ Translates to Anthropic format
Your Code (Anthropic SDK)
```

## 🎯 Use Cases

| Scenario | Solution | Example File |
|----------|----------|--------------|
| New app, multi-provider | Forward translation | `anthropic_example.py` |
| Existing Anthropic code | Reverse translation | `reverse_translation_example.py` |
| Migration from Anthropic | Proxy with Anthropic format | `anthropic_to_proxy_migration.py` |
| Production deployment | Proxy with load balancing | `complete_anthropic_proxy_example.py` |
| Quick test | Direct library usage | `anthropic_quickstart.py` |

## 🔧 Supported Features

| Feature | Forward (OpenAI→Anthropic) | Reverse (Anthropic→OpenAI) |
|---------|---------------------------|---------------------------|
| Basic Completion | ✅ | ✅ |
| Streaming | ✅ | ✅ |
| Async | ✅ | ✅ |
| System Prompts | ✅ | ✅ |
| Tool Calling | ✅ | ✅ |
| Vision | ✅ | ✅ |
| Prompt Caching | ✅ | ✅ |
| Extended Thinking | ✅ | ✅ |
| Load Balancing | ✅ | ✅ |
| Fallbacks | ✅ | ✅ |

## 📖 Available Models

```python
# Claude 3.5 Sonnet - Best balance
"anthropic/claude-3-5-sonnet-20241022"

# Claude 3.5 Haiku - Fastest & cheapest
"anthropic/claude-3-5-haiku-20241022"

# Claude 3 Opus - Most capable
"anthropic/claude-3-opus-20240229"

# Claude Sonnet 4 - Extended thinking
"anthropic/claude-3-7-sonnet-20250219"
```

## 🎓 Learning Path

1. **Start Simple**: Run `anthropic_quickstart.py`
2. **Explore Features**: Try examples in `anthropic_example.py`
3. **Understand Translation**: Read `anthropic_code_flow.md`
4. **Learn Reverse Translation**: Check `reverse_translation_example.py`
5. **Deploy with Proxy**: Use `complete_anthropic_proxy_example.py`
6. **Deep Dive**: Read `ANTHROPIC_GUIDE.md`

## 💡 Key Concepts

### Bidirectional Translation

LiteLLM supports translation in **both directions**:

1. **Forward**: Your code uses OpenAI format → LiteLLM translates → Anthropic API
2. **Reverse**: Your code uses Anthropic format → LiteLLM translates → Any provider

This gives you maximum flexibility to:
- Use whichever format you prefer
- Migrate between providers easily
- Keep existing code unchanged
- Add proxy features without rewrites

### Why Use LiteLLM?

**For New Projects:**
- Unified interface across 100+ providers
- Easy provider switching
- Built-in retry, fallback, load balancing

**For Existing Anthropic Code:**
- Keep your Anthropic SDK code
- Route to different providers via proxy
- Add cost tracking, rate limiting
- Zero code changes (just 2 lines)

## 🔗 Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)

## 🆘 Need Help?

1. Check the examples in this directory
2. Read the troubleshooting section in `ANTHROPIC_GUIDE.md`
3. Join the [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)
4. Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)

## 📝 Quick Examples

### Forward Translation (OpenAI Format)
```python
from litellm import completion

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Reverse Translation (Anthropic Format)
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000/anthropic"
)

message = client.messages.create(
    model="gpt-4",  # Routes to OpenAI via proxy!
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

---

**Ready to get started?** Run `python anthropic_quickstart.py` or explore the examples above!
