# Configuration Files Index

Configuration files for LiteLLM proxy with Anthropic integration.

## 📄 Configuration Files

### General Configuration

- **[proxy_config.yaml](proxy_config.yaml)** - General proxy configuration
  - Multiple model configurations
  - Load balancing examples
  - Fallback configuration
  - Authentication setup
  - Rate limiting
  - Cost tracking
  - Commented examples for all features

### Provider-Specific Configuration

- **[github_copilot_example.yaml](github_copilot_example.yaml)** - GitHub Copilot configuration
  - GitHub Copilot models (GPT-4, GPT-3.5)
  - OAuth2 authentication (automatic)
  - No API key needed
  - Verbose logging enabled

## 🚀 How to Use

### Start Proxy with Config

```bash
# General configuration
litellm --config proxy_config.yaml

# GitHub Copilot
litellm --config github_copilot_example.yaml

# Custom port
litellm --config proxy_config.yaml --port 4001
```

### Test Configuration

```bash
# Check if proxy is running
curl http://localhost:4000/health

# List available models
curl http://localhost:4000/models

# Test completion
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## 📋 Configuration Options

### Basic Structure

```yaml
model_list:
  - model_name: <alias>
    litellm_params:
      model: <provider>/<model>
      api_key: <key or os.environ/VAR>

general_settings:
  master_key: <proxy auth key>
  database_url: <postgres url>

litellm_settings:
  set_verbose: true
  drop_params: true
```

### Common Configurations

#### Single Model (Simplest)

```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
```

#### Multiple Models

```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: gpt4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

#### Load Balancing

```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY_1
  
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY_2

router_settings:
  routing_strategy: simple-shuffle
```

#### Fallbacks

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

fallbacks:
  - claude-sonnet: ["claude-haiku"]
```

#### Authentication

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://user:pass@localhost/litellm
```

#### Budget & Rate Limiting

```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  max_budget: 1000  # $1000 max
  budget_duration: 30d
```

## 🎯 Configuration by Use Case

### Development/Testing

```yaml
# No authentication, single model
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  set_verbose: true
```

### Production (Basic)

```yaml
# With authentication
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY

litellm_settings:
  set_verbose: false
```

### Production (Advanced)

```yaml
# Multiple models, fallbacks, auth, budgets
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

fallbacks:
  - claude-sonnet: ["claude-haiku"]

router_settings:
  routing_strategy: latency-based-routing
  num_retries: 3

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  max_budget: 1000
  budget_duration: 30d

litellm_settings:
  set_verbose: false
  drop_params: true
```

## 🔧 Environment Variables

### Required

```bash
# For Anthropic models
export ANTHROPIC_API_KEY="sk-ant-..."

# For OpenAI models
export OPENAI_API_KEY="sk-..."

# For Azure models
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://..."
```

### Optional

```bash
# For proxy authentication
export LITELLM_MASTER_KEY="sk-1234"

# For database (virtual keys, cost tracking)
export DATABASE_URL="postgresql://..."
```

## 📖 Configuration Reference

### model_list
- `model_name`: Alias for the model
- `litellm_params.model`: Provider and model name
- `litellm_params.api_key`: API key or environment variable

### general_settings
- `master_key`: Proxy authentication key
- `database_url`: PostgreSQL database for virtual keys
- `max_budget`: Maximum spend limit
- `budget_duration`: Budget reset period

### router_settings
- `routing_strategy`: How to route requests
- `num_retries`: Number of retries on failure
- `timeout`: Request timeout

### litellm_settings
- `set_verbose`: Enable detailed logging
- `drop_params`: Drop unsupported parameters
- `success_callback`: Logging integrations
- `failure_callback`: Error logging

### fallbacks
- Map primary model to fallback models
- Format: `primary: ["fallback1", "fallback2"]`

## 🆘 Troubleshooting

### Config not loading
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('proxy_config.yaml'))"
```

### Environment variables not found
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Model not found
- Check `model_name` in config
- Verify API key is correct
- Check provider name format

### Authentication errors
- Ensure `master_key` is set if using auth
- Pass correct key in requests
- Check database connection if using virtual keys

## 💡 Tips

- **Start simple**: Use basic config first
- **Test locally**: Use development config for testing
- **Use environment variables**: Never hardcode API keys
- **Enable verbose logging**: Helps debug issues
- **Comment your config**: Document your choices

## 🔗 Related Documentation

- [../docs/ENDPOINT_AVAILABILITY.md](../docs/ENDPOINT_AVAILABILITY.md) - Endpoint details
- [../docs/AUTHENTICATION_GUIDE.md](../docs/AUTHENTICATION_GUIDE.md) - Authentication setup
- [../docs/ROUTING_LOGIC.md](../docs/ROUTING_LOGIC.md) - How routing works
