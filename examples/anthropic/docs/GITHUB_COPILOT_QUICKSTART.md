# GitHub Copilot with Anthropic SDK Format - Quick Start

## What This Does

Use the **Anthropic SDK** (with its native format) to call **GitHub Copilot's GPT-4 models**. LiteLLM automatically translates between formats.

## Why This is Useful

- Keep your Anthropic SDK code unchanged
- Use GitHub Copilot models instead of Claude
- No code rewrite needed - just change the model name
- GitHub Copilot uses OAuth2 (no API key management!)

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install anthropic litellm[proxy]
```

### Step 2: Start LiteLLM Proxy

```bash
# Navigate to examples/anthropic directory
cd examples/anthropic

# Start the proxy
litellm --config github_copilot_example.yaml
```

You should see:
```
INFO: Proxy running on http://0.0.0.0:4000
```

### Step 3: Run the Test

In a new terminal:

```bash
python test_github_copilot.py
```

## What You'll See

```
================================================================================
GitHub Copilot + Anthropic SDK Format Test
================================================================================
✅ LiteLLM proxy is running

================================================================================
TRANSLATION DETAILS
================================================================================

When you call:
    client.messages.create(
        model="copilot-gpt4",
        system="You are helpful",
        messages=[{"role": "user", "content": "Hello"}]
    )

Here's what happens:

1. Request Format (Anthropic)
2. LiteLLM Proxy Detects: Provider is NOT Anthropic → Translation required
3. Translation (Anthropic → OpenAI)
4. Call GitHub Copilot API (OpenAI format)
5. Receive Response (OpenAI format)
6. Translation (OpenAI → Anthropic)
7. Your Code Receives Anthropic Format!

================================================================================
RUNNING TESTS
================================================================================

================================================================================
TEST 1: Basic Completion
================================================================================

📤 Sending request in Anthropic format:
   Model: copilot-gpt4
   Format: Anthropic /v1/messages
   Content: 'Explain what GitHub Copilot is in one sentence'

📥 Response received in Anthropic format:
   ID: chatcmpl-123...
   Model: gpt-4
   Type: message
   Role: assistant
   Stop Reason: stop

   Content: GitHub Copilot is an AI-powered code completion tool...

   Usage:
   - Input tokens: 15
   - Output tokens: 25

✅ SUCCESS: Translation worked!
   Anthropic format → OpenAI format → GitHub Copilot → Anthropic format
```

## Your Own Code

Once the proxy is running, use this in your application:

```python
import anthropic

# Point to LiteLLM proxy
client = anthropic.Anthropic(
    api_key="anything",  # No auth needed
    base_url="http://localhost:4000"
)

# Use Anthropic format with GitHub Copilot!
message = client.messages.create(
    model="copilot-gpt4",  # GitHub Copilot model
    max_tokens=1024,
    system="You are a helpful coding assistant",
    messages=[
        {"role": "user", "content": "Write a Python function to reverse a string"}
    ]
)

print(message.content[0].text)
```

## Available Models

The config includes three GitHub Copilot models:

```python
# GPT-4 (most capable)
model="copilot-gpt4"

# GPT-4 Turbo (faster)
model="copilot-gpt4-turbo"

# GPT-3.5 Turbo (fastest, cheapest)
model="copilot-gpt35"
```

## Features That Work

All Anthropic SDK features work with GitHub Copilot:

✅ **Basic completion**
```python
message = client.messages.create(
    model="copilot-gpt4",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
)
```

✅ **System prompts**
```python
message = client.messages.create(
    model="copilot-gpt4",
    max_tokens=100,
    system="You are a helpful assistant",  # ← Translated automatically
    messages=[{"role": "user", "content": "Hello"}]
)
```

✅ **Streaming**
```python
with client.messages.stream(
    model="copilot-gpt4",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

✅ **Multi-turn conversations**
```python
message = client.messages.create(
    model="copilot-gpt4",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What about 2+3?"}
    ]
)
```

## Authentication

GitHub Copilot uses **OAuth2** authentication, which is handled automatically by LiteLLM. You don't need to:
- Set any API keys
- Manage tokens
- Configure authentication

LiteLLM handles it all!

## Troubleshooting

### Proxy won't start

**Error:** `Address already in use`

**Solution:** Another service is using port 4000
```bash
# Use a different port
litellm --config github_copilot_example.yaml --port 4001

# Update your code
base_url="http://localhost:4001"
```

### Connection refused

**Error:** `Connection refused to localhost:4000`

**Solution:** Proxy isn't running
```bash
# Start the proxy in another terminal
litellm --config github_copilot_example.yaml
```

### GitHub Copilot authentication failed

**Error:** `Authentication error with GitHub Copilot`

**Solution:** Check GitHub Copilot access
1. Ensure you have GitHub Copilot subscription
2. Check your GitHub Copilot settings
3. Try authenticating with GitHub CLI: `gh auth login`

### Model not found

**Error:** `Model 'copilot-gpt4' not found`

**Solution:** Check config file
```bash
# Verify config is correct
cat github_copilot_example.yaml

# Restart proxy
litellm --config github_copilot_example.yaml
```

## How Translation Works

### Request Translation (Anthropic → OpenAI)

**Your code (Anthropic format):**
```python
{
    "model": "copilot-gpt4",
    "system": "You are helpful",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
}
```

**What LiteLLM sends to GitHub Copilot (OpenAI format):**
```python
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 1024
}
```

### Response Translation (OpenAI → Anthropic)

**GitHub Copilot returns (OpenAI format):**
```python
{
    "id": "chatcmpl-123",
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help?"
        }
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8}
}
```

**What you receive (Anthropic format):**
```python
{
    "id": "chatcmpl-123",
    "type": "message",
    "role": "assistant",
    "content": [{
        "type": "text",
        "text": "Hello! How can I help?"
    }],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8
    }
}
```

## Next Steps

1. **Run the test:** `python test_github_copilot.py`
2. **Try in your code:** Use the example above
3. **Explore features:** Test streaming, system prompts, etc.
4. **Add more models:** Edit `github_copilot_example.yaml`

## Benefits

✅ **No code changes** - Keep using Anthropic SDK
✅ **No API key management** - OAuth2 handled automatically
✅ **Same format** - Anthropic format works with GitHub Copilot
✅ **Easy switching** - Change model name to switch providers
✅ **All features work** - Streaming, system prompts, multi-turn

## Summary

You can now use the Anthropic SDK with GitHub Copilot models! LiteLLM handles all the translation automatically. Just:

1. Start proxy: `litellm --config github_copilot_example.yaml`
2. Point Anthropic SDK to proxy: `base_url="http://localhost:4000"`
3. Use GitHub Copilot models: `model="copilot-gpt4"`

That's it! 🎉
