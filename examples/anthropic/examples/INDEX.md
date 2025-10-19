# Examples Index

Runnable code examples for LiteLLM + Anthropic integration.

## 📝 Example Files

### Quick Start

- **[anthropic_quickstart.py](anthropic_quickstart.py)** - Minimal 10-line example
  - Simplest way to get started
  - Basic completion
  - No configuration needed

### Comprehensive Examples

- **[anthropic_example.py](anthropic_example.py)** - 13 detailed examples
  1. Basic completion
  2. Streaming
  3. Async operations
  4. Multi-turn conversations
  5. System prompts
  6. Tool/function calling
  7. Vision (image analysis)
  8. Advanced parameters
  9. Prompt caching
  10. Extended thinking (Claude Sonnet 4)
  11. Error handling
  12. Model comparison
  13. Load balancing with Router

### Migration & Proxy

- **[anthropic_to_proxy_migration.py](anthropic_to_proxy_migration.py)** - Migration guide
  - Before/after comparison
  - Original Anthropic SDK vs LiteLLM Proxy
  - Benefits comparison
  - Migration checklist

- **[complete_anthropic_proxy_example.py](complete_anthropic_proxy_example.py)** - Real-world examples
  - Chatbot implementation
  - Batch processing
  - Tool calling
  - Interactive menu

### Reverse Translation

- **[reverse_translation_example.py](reverse_translation_example.py)** - Anthropic format → Any provider
  - Use Anthropic SDK with OpenAI
  - Use Anthropic SDK with Azure
  - Translation flow explanation
  - Code comparison

### Testing

- **[test_github_copilot.py](test_github_copilot.py)** - GitHub Copilot integration test
  - 5 comprehensive tests
  - Basic completion
  - System prompts
  - Streaming
  - Multi-turn conversations
  - Different models

- **[no_code_change_example.py](no_code_change_example.py)** - Zero code changes demo
  - Keep existing Anthropic code
  - Just change 2 lines
  - Route to any provider

## 🚀 How to Run

### Prerequisites

```bash
# Install dependencies
pip install litellm anthropic

# For proxy examples
pip install 'litellm[proxy]'
```

### Running Examples

```bash
# Quick start (no proxy needed)
python anthropic_quickstart.py

# Comprehensive examples (no proxy needed)
python anthropic_example.py

# Proxy examples (start proxy first)
# Terminal 1:
litellm --config ../configs/proxy_config.yaml

# Terminal 2:
python complete_anthropic_proxy_example.py

# GitHub Copilot test
# Terminal 1:
litellm --config ../configs/github_copilot_example.yaml

# Terminal 2:
python test_github_copilot.py
```

## 🎯 Choose by Use Case

### I want to...

**Get started quickly**
→ Run [anthropic_quickstart.py](anthropic_quickstart.py)

**See all features**
→ Run [anthropic_example.py](anthropic_example.py)

**Migrate existing code**
→ Check [anthropic_to_proxy_migration.py](anthropic_to_proxy_migration.py)

**Use Anthropic SDK with other providers**
→ Try [reverse_translation_example.py](reverse_translation_example.py)

**Test GitHub Copilot**
→ Run [test_github_copilot.py](test_github_copilot.py)

**See real-world usage**
→ Explore [complete_anthropic_proxy_example.py](complete_anthropic_proxy_example.py)

## 📊 Example Comparison

| Example | Proxy Required | Features | Complexity |
|---------|---------------|----------|------------|
| anthropic_quickstart.py | ❌ No | Basic | ⭐ Simple |
| anthropic_example.py | ❌ No | All | ⭐⭐ Medium |
| anthropic_to_proxy_migration.py | ✅ Yes | Comparison | ⭐⭐ Medium |
| complete_anthropic_proxy_example.py | ✅ Yes | Real-world | ⭐⭐⭐ Advanced |
| reverse_translation_example.py | ✅ Yes | Translation | ⭐⭐ Medium |
| test_github_copilot.py | ✅ Yes | Testing | ⭐⭐ Medium |
| no_code_change_example.py | ✅ Yes | Migration | ⭐ Simple |

## 🔧 Configuration

Most examples work without configuration. For proxy examples, see:
- [../configs/proxy_config.yaml](../configs/proxy_config.yaml)
- [../configs/github_copilot_example.yaml](../configs/github_copilot_example.yaml)

## 📖 Learning Path

### Beginner
1. [anthropic_quickstart.py](anthropic_quickstart.py) - Start here
2. [anthropic_example.py](anthropic_example.py) - Explore features
3. [test_github_copilot.py](test_github_copilot.py) - Try proxy

### Intermediate
1. [anthropic_to_proxy_migration.py](anthropic_to_proxy_migration.py) - Understand migration
2. [reverse_translation_example.py](reverse_translation_example.py) - Learn translation
3. [complete_anthropic_proxy_example.py](complete_anthropic_proxy_example.py) - Real usage

### Advanced
1. Study all examples
2. Modify for your use case
3. Deploy to production

## 💡 Tips

- **Start simple**: Begin with `anthropic_quickstart.py`
- **Read comments**: Each example has detailed comments
- **Try modifications**: Change parameters and see what happens
- **Check logs**: Enable verbose mode to see what's happening
- **Combine examples**: Mix and match code from different examples

## 🆘 Troubleshooting

### Import errors
```bash
pip install litellm anthropic
```

### Proxy connection errors
```bash
# Make sure proxy is running
litellm --config ../configs/proxy_config.yaml
```

### API key errors
```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Model not found
Check the config file and ensure the model is defined.
