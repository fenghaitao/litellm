# Running GitHub Copilot Example

## Quick Start

### Terminal 1: Start Proxy (No Auth)

```bash
cd examples/anthropic

# Start proxy without authentication
litellm --model github_copilot/gpt-4
```

This starts the proxy with:
- No authentication required
- GitHub Copilot GPT-4 model available
- Endpoint: `http://localhost:4000`

### Terminal 2: Run Test

```bash
cd examples/anthropic
python test_github_copilot.py
```

## Alternative: With Config File

If you want multiple models and more control:

### Terminal 1: Start with Config

```bash
cd examples/anthropic
litellm --config github_copilot_example.yaml
```

### Terminal 2: Run Test

```bash
python test_github_copilot.py
```

## What You Should See

### Proxy Output (Terminal 1):

```
LiteLLM: Proxy running on http://0.0.0.0:4000
```

### Test Output (Terminal 2):

```
================================================================================
GitHub Copilot + Anthropic SDK Format Test
================================================================================
✅ LiteLLM proxy is running

================================================================================
TEST 1: Basic Completion
================================================================================

📤 Sending request in Anthropic format:
   Model: copilot-gpt4
   Format: Anthropic /v1/messages
   Content: 'Explain what GitHub Copilot is in one sentence'

📥 Response received in Anthropic format:
   ID: chatcmpl-...
   Model: gpt-4
   Type: message
   Role: assistant
   
   Content: GitHub Copilot is an AI-powered code completion tool...
   
✅ SUCCESS: Translation worked!
```

## Troubleshooting

### Error: "No api key passed in" (401 Unauthorized)

**Problem:** Proxy requires authentication

**Solution 1:** Start proxy without config (simplest)
```bash
litellm --model github_copilot/gpt-4
```

**Solution 2:** Disable auth in config
Edit `github_copilot_example.yaml` and ensure `general_settings.master_key` is commented out:
```yaml
# general_settings:
#   master_key: os.environ/LITELLM_MASTER_KEY
```

**Solution 3:** Set master key and update test
```bash
# Terminal 1
export LITELLM_MASTER_KEY="sk-1234"
litellm --config github_copilot_example.yaml

# Terminal 2 - Update test script to use the key
# Change: api_key="anything"
# To: api_key="sk-1234"
```

### Error: "Connection refused"

**Problem:** Proxy not running

**Solution:** Start proxy in Terminal 1
```bash
litellm --model github_copilot/gpt-4
```

### Error: "Model not found"

**Problem:** Model name mismatch

**Solution:** Check model name in test matches proxy config
- If using `--model github_copilot/gpt-4`, use `model="gpt-4"` in test
- If using config file, use `model="copilot-gpt4"` in test

## Simple Test Without Script

You can test directly with curl:

```bash
curl http://localhost:4000/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "gpt-4",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

Or with Python:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="anything",
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="gpt-4",  # or "copilot-gpt4" if using config
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

## Summary

**Simplest way:**
```bash
# Terminal 1
litellm --model github_copilot/gpt-4

# Terminal 2
python test_github_copilot.py
```

That's it! The proxy handles OAuth2 authentication with GitHub Copilot automatically.
