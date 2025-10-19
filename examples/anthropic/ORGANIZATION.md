# Directory Organization

The `examples/anthropic` directory is now organized for easy navigation and use.

## 📁 Structure

```
examples/anthropic/
├── README.md                    # Main entry point - START HERE
├── ORGANIZATION.md              # This file - directory structure
│
├── docs/                        # 📚 Documentation
│   ├── INDEX.md                # Documentation index
│   ├── SUMMARY.md              # Complete overview
│   ├── ANTHROPIC_GUIDE.md      # Integration guide
│   ├── AUTHENTICATION_GUIDE.md # Authentication details
│   ├── ENDPOINT_AVAILABILITY.md # Proxy endpoints
│   ├── ROUTING_LOGIC.md        # Routing decisions
│   ├── TRANSLATION_TRIGGER.md  # Translation details
│   ├── anthropic_code_flow.md  # Internal code flow
│   ├── GITHUB_COPILOT_QUICKSTART.md # GitHub Copilot guide
│   └── RUN_GITHUB_COPILOT.md   # GitHub Copilot setup
│
├── examples/                    # 💻 Code Examples
│   ├── INDEX.md                # Examples index
│   ├── anthropic_quickstart.py # Quick start (10 lines)
│   ├── anthropic_example.py    # 13 comprehensive examples
│   ├── anthropic_to_proxy_migration.py # Migration guide
│   ├── complete_anthropic_proxy_example.py # Real-world examples
│   ├── reverse_translation_example.py # Reverse translation
│   ├── test_github_copilot.py  # GitHub Copilot test
│   └── no_code_change_example.py # No code change demo
│
├── configs/                     # ⚙️ Configuration Files
│   ├── INDEX.md                # Config index
│   ├── proxy_config.yaml       # General proxy config
│   └── github_copilot_example.yaml # GitHub Copilot config
│
└── server/                      # 🔧 Server (external project)
    └── ...                      # Anthropic server implementation
```

## 🎯 Quick Navigation

### I want to...

**Get started**
→ Read [README.md](README.md)

**Understand everything**
→ Read [docs/SUMMARY.md](docs/SUMMARY.md)

**See code examples**
→ Browse [examples/](examples/) or check [examples/INDEX.md](examples/INDEX.md)

**Configure the proxy**
→ Check [configs/](configs/) or see [configs/INDEX.md](configs/INDEX.md)

**Learn specific topics**
→ Browse [docs/](docs/) or check [docs/INDEX.md](docs/INDEX.md)

## 📚 Documentation (docs/)

All documentation files organized by topic:

- **Getting Started**: SUMMARY.md, ANTHROPIC_GUIDE.md
- **Authentication**: AUTHENTICATION_GUIDE.md, ENDPOINT_AVAILABILITY.md
- **How It Works**: ROUTING_LOGIC.md, TRANSLATION_TRIGGER.md, anthropic_code_flow.md
- **GitHub Copilot**: GITHUB_COPILOT_QUICKSTART.md, RUN_GITHUB_COPILOT.md

Each subdirectory has an INDEX.md for easy navigation.

## 💻 Examples (examples/)

All runnable code examples:

- **Quick Start**: anthropic_quickstart.py (10 lines)
- **Comprehensive**: anthropic_example.py (13 examples)
- **Migration**: anthropic_to_proxy_migration.py, no_code_change_example.py
- **Advanced**: complete_anthropic_proxy_example.py, reverse_translation_example.py
- **Testing**: test_github_copilot.py

See [examples/INDEX.md](examples/INDEX.md) for details on each file.

## ⚙️ Configurations (configs/)

Configuration files for different use cases:

- **General**: proxy_config.yaml (with all options commented)
- **GitHub Copilot**: github_copilot_example.yaml (OAuth2, no API key)

See [configs/INDEX.md](configs/INDEX.md) for configuration reference.

## 🚀 Getting Started

### 1. Read the README
```bash
cat README.md
```

### 2. Try Quick Start
```bash
python examples/anthropic_quickstart.py
```

### 3. Explore Documentation
```bash
# Read overview
cat docs/SUMMARY.md

# Or browse all docs
ls docs/
```

### 4. Run Examples
```bash
# See all examples
ls examples/

# Run comprehensive examples
python examples/anthropic_example.py
```

### 5. Configure Proxy
```bash
# Start with config
litellm --config configs/proxy_config.yaml

# Or GitHub Copilot
litellm --config configs/github_copilot_example.yaml
```

## 📖 Learning Path

### Beginner Path
1. [README.md](README.md) - Overview
2. [examples/anthropic_quickstart.py](examples/anthropic_quickstart.py) - Try it
3. [docs/ANTHROPIC_GUIDE.md](docs/ANTHROPIC_GUIDE.md) - Learn more
4. [examples/anthropic_example.py](examples/anthropic_example.py) - Explore features

### Intermediate Path
1. [docs/ROUTING_LOGIC.md](docs/ROUTING_LOGIC.md) - Understand routing
2. [examples/reverse_translation_example.py](examples/reverse_translation_example.py) - Try reverse translation
3. [docs/AUTHENTICATION_GUIDE.md](docs/AUTHENTICATION_GUIDE.md) - Set up auth
4. [examples/complete_anthropic_proxy_example.py](examples/complete_anthropic_proxy_example.py) - Real usage

### Advanced Path
1. [docs/TRANSLATION_TRIGGER.md](docs/TRANSLATION_TRIGGER.md) - Deep dive
2. [docs/anthropic_code_flow.md](docs/anthropic_code_flow.md) - Understand internals
3. [configs/proxy_config.yaml](configs/proxy_config.yaml) - Advanced config
4. Deploy to production

## 🔍 Find What You Need

### By File Type

- **Documentation**: `docs/*.md`
- **Code Examples**: `examples/*.py`
- **Configuration**: `configs/*.yaml`
- **Indexes**: `*/INDEX.md`

### By Topic

- **Authentication**: docs/AUTHENTICATION_GUIDE.md
- **Translation**: docs/ROUTING_LOGIC.md, docs/TRANSLATION_TRIGGER.md
- **GitHub Copilot**: docs/GITHUB_COPILOT_QUICKSTART.md, examples/test_github_copilot.py
- **Migration**: examples/anthropic_to_proxy_migration.py
- **Configuration**: configs/INDEX.md

### By Use Case

- **New project**: examples/anthropic_quickstart.py
- **Existing Anthropic code**: examples/reverse_translation_example.py
- **Production deployment**: docs/SUMMARY.md, configs/proxy_config.yaml
- **Testing**: examples/test_github_copilot.py

## 💡 Tips

- **Start with README.md** - It has links to everything
- **Use INDEX.md files** - Each directory has one
- **Read comments in examples** - They explain what's happening
- **Check configs/INDEX.md** - For configuration reference
- **Browse docs/** - For detailed explanations

## 🆘 Need Help?

1. Check [README.md](README.md) for overview
2. Browse [docs/INDEX.md](docs/INDEX.md) for documentation
3. See [examples/INDEX.md](examples/INDEX.md) for code examples
4. Check [configs/INDEX.md](configs/INDEX.md) for configuration
5. Join [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)

## 📝 File Naming Convention

- **UPPERCASE.md**: Documentation files
- **lowercase.py**: Code examples
- **lowercase.yaml**: Configuration files
- **INDEX.md**: Directory indexes

## 🔗 External Links

- [LiteLLM Docs](https://docs.litellm.ai/)
- [Anthropic Docs](https://docs.anthropic.com/)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [LiteLLM Discord](https://discord.gg/wuPM9dRgDw)

---

**Everything is organized and ready to use!** Start with [README.md](README.md) 🚀
