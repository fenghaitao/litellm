# LiteLLM + Anthropic Examples

Complete examples and documentation for using LiteLLM with Anthropic's Claude models, including **bidirectional translation** between OpenAI and Anthropic formats.

## 📁 Directory Structure

```
examples/anthropic/
├── README.md                    # This file - start here
├── docs/                        # Documentation
│   ├── SUMMARY.md              # Complete overview
│   ├── ANTHROPIC_GUIDE.md      # Integration guide
│   ├── AUTHENTICATION_GUIDE.md # Authentication details
│   ├── ENDPOINT_AVAILABILITY.md # Proxy endpoints
│   ├── ROUTING_LOGIC.md        # How routing works
│   ├── TRANSLATION_TRIGGER.md  # Translation details
│   ├── anthropic_code_flow.md  # Internal code flow
│   ├── GITHUB_COPILOT_QUICKSTART.md # GitHub Copilot guide
│   └── RUN_GITHUB_COPILOT.md   # GitHub Copilot setup
├── examples/                    # Code examples
│   ├── anthropic_quickstart.py # 10-line quick start
│   ├── anthropic_example.py    # 13 comprehensive examples
│   ├── anthropic_to_proxy_migration.py # Migration guide
│   ├── complete_anthropic_proxy_example.py # Full examples
│   ├── reverse_translation_example.py # Reverse translation
│   ├── test_github_copilot.py  # GitHub Copilot test
│   └── no_code_change_example.py # No code change demo
└── configs/                     # Configuration files
    ├── proxy_config.yaml       # General proxy config
    └── github_copilot_example.yaml # GitHub Copilot config
```

## 🚀 Quick Start

### Option 1: Library (Forward Translation)

Use OpenAI format with Anthropic models:

```bash
# Install
pip install litellm

# Run
python examples/anthropic_quickstart.py
```

### Option 2: Proxy (Reverse Translation)

Use Anthropic format with any provider:

```bash
# Install
pip install 'litellm[proxy]'

# Start proxy
litellm --config configs/proxy_config.yaml

# Test
python examples/test_github_copilot.py
```

## 📚 Documentation

### Getting Started
- **[SUMMARY.md](docs/SUMMARY.md)** - Complete overview of everything
- **[ANTHROPIC_GUIDE.md](docs/ANTHROPIC_GUIDE.md)** - Integration guide with examples
- **[anthropic_code_flow.md](docs/anthropic_code_flow.md)** - How it works internally

### Authentication & Setup
- **[AUTHENTICATION_GUIDE.md](docs/AUTHENTICATION_GUIDE.md)** - How to pass API keys
- **[ENDPOINT_AVAILABILITY.md](docs/ENDPOINT_AVAILABILITY.md)** - Available endpoints

### Advanced Topics
- **[ROUTING_LOGIC.md](docs/ROUTING_LOGIC.md)** - How LiteLLM decides routing
- **[TRANSLATION_TRIGGER.md](docs/TRANSLATION_TRIGGER.md)** - When translation happens

### GitHub Copilot
- **[GITHUB_COPILOT_QUICKSTART.md](docs/GITHUB_COPILOT_QUICKSTART.md)** - Use Anthropic SDK with GitHub Copilot
- **[RUN_GITHUB_COPILOT.md](docs/RUN_GITHUB_COPILOT.md)** - Setup guide

## 💻 Code Examples

### Basic Examples
- **[anthropic_quickstart.py](examples/anthropic_quickstart.py)** - Minimal 10-line example
- **[anthropic_example.py](examples/anthropic_example.py)** - 13 comprehensive examples

### Migration & Proxy
- **[anthropic_to_proxy_migration.py](examples/anthropic_to_proxy_migration.py)** - Before/after comparison
- **[complete_anthropic_proxy_example.py](examples/complete_anthropic_proxy_example.py)** - Real-world examples
- **[reverse_translation_example.py](examples/reverse_translation_example.py)** - Anthropic format → Any provider

### Testing
- **[test_github_copilot.py](examples/test_github_copilot.py)** - Test GitHub Copilot integration
- **[no_code_change_example.py](examples/no_code_change_example.py)** - Zero code changes demo

## ⚙️ Configuration Files

- **[proxy_config.yaml](configs/proxy_config.yaml)** - General proxy configuration with examples
- **[github_copilot_example.yaml](configs/github_copilot_example.yaml)** - GitHub Copilot specific config

## 🎯 Use Cases

| Scenario | Documentation | Example | Config |
|----------|--------------|---------|--------|
| New app, multi-provider | [ANTHROPIC_GUIDE.md](docs/ANTHROPIC_GUIDE.md) | [anthropic_example.py](examples/anthropic_example.py) | - |
| Existing Anthropic code | [SUMMARY.md](docs/SUMMARY.md) | [reverse_translation_example.py](examples/reverse_translation_example.py) | [proxy_config.yaml](configs/proxy_config.yaml) |
| GitHub Copilot | [GITHUB_COPILOT_QUICKSTART.md](docs/GITHUB_COPILOT_QUICKSTART.md) | [test_github_copilot.py](examples/test_github_copilot.py) | [github_copilot_example.yaml](configs/github_copilot_example.yaml) |
| Migration from Anthropic | [ANTHROPIC_GUIDE.md](docs/ANTHROPIC_GUIDE.md) | [anthropic_to_proxy_migration.py](examples/anthropic_to_proxy_migration.py) | [proxy_config.yaml](configs/proxy_config.yaml) |
| Production deployment | [SUMMARY.md](docs/SUMMARY.md) | [complete_anthropic_proxy_example.py](examples/complete_anthropic_proxy_example.py) | [proxy_config.yaml](configs/proxy_config.yaml) |

## 🔄 Two Ways to Use LiteLLM

### Forward Translation (OpenAI → Anthropic)

Use OpenAI format, LiteLLM translates to Anthropic:

```python
from litellm import completion

response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Reverse Translation (Anthropic → Any Provider)

Use Anthropic format, LiteLLM translates to any provider:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-proxy-key",
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="gpt-4",  # Routes to OpenAI!
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## 📖 Learning Path

1. **Start Simple**: Run [anthropic_quickstart.py](examples/anthropic_quickstart.py)
2. **Explore Features**: Try [anthropic_example.py](examples/anthropic_example.py)
3. **Understand Translation**: Read [ROUTING_LOGIC.md](docs/ROUTING_LOGIC.md)
4. **Learn Reverse Translation**: Check [reverse_translation_example.py](examples/reverse_translation_example.py)
5. **Deploy with Proxy**: Use [complete_anthropic_proxy_example.py](examples/complete_anthropic_proxy_example.py)
6. **Deep Dive**: Read [SUMMARY.md](docs/SUMMARY.md)

## 🔗 External Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Anthropic Documentation](https://docs.anthropic.com/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)

## 🆘 Need Help?

1. Check the [docs/](docs/) directory for detailed guides
2. Run the [examples/](examples/) to see working code
3. Read [SUMMARY.md](docs/SUMMARY.md) for complete overview
4. Join [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)
5. Open an issue on [GitHub](https://github.com/BerriAI/litellm/issues)

## 📝 Quick Reference

### Installation
```bash
# Library only
pip install litellm

# With proxy
pip install 'litellm[proxy]'

# With Anthropic SDK
pip install anthropic
```

### Start Proxy
```bash
# Basic
litellm --model anthropic/claude-3-5-sonnet-20241022

# With config
litellm --config configs/proxy_config.yaml

# GitHub Copilot
litellm --config configs/github_copilot_example.yaml
```

### Test
```bash
# Quick start
python examples/anthropic_quickstart.py

# Comprehensive examples
python examples/anthropic_example.py

# GitHub Copilot
python examples/test_github_copilot.py
```

---

**Ready to get started?** Pick a use case above and follow the links! 🚀
