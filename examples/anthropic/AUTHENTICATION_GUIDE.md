# Authentication with LiteLLM Proxy and Anthropic SDK

## Quick Answer

When using the Anthropic SDK with LiteLLM proxy, pass the `LITELLM_MASTER_KEY` as the `api_key` parameter:

```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",  # Your LITELLM_MASTER_KEY here
    base_url="http://localhost:4000"
)
```

The Anthropic SDK will send this as the `x-api-key` header, which LiteLLM proxy uses for authentication.

## Complete Authentication Guide

### Scenario 1: No Authentication (Development/Testing)

**Start proxy without master key:**
```bash
litellm --model anthropic/claude-3-5-sonnet-20241022
```

**Use any API key (or "anything"):**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="anything",  # Any value works
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**With cURL:**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: anything" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

### Scenario 2: Master Key Authentication (Recommended)

**Set master key and start proxy:**
```bash
export LITELLM_MASTER_KEY="sk-1234"
litellm --model anthropic/claude-3-5-sonnet-20241022
```

**Pass master key as api_key:**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",  # Your LITELLM_MASTER_KEY
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**With cURL:**
```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: sk-1234" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

### Scenario 3: Master Key via Config File

**Create config.yaml:**
```yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY  # Read from environment
```

**Set environment variable and start:**
```bash
export LITELLM_MASTER_KEY="sk-1234"
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --config config.yaml
```

**Use in code:**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-1234",  # Your LITELLM_MASTER_KEY
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude",  # Model name from config
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

### Scenario 4: Virtual Keys (Production)

**Step 1: Start proxy with master key and database:**
```yaml
# config.yaml
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://user:pass@localhost/litellm
```

```bash
export LITELLM_MASTER_KEY="sk-master-1234"
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --config config.yaml
```

**Step 2: Generate a virtual key using master key:**
```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["claude"],
    "max_budget": 100,
    "duration": "30d"
  }'
```

**Response:**
```json
{
  "key": "sk-litellm-abc123...",
  "expires": "2024-12-01T00:00:00"
}
```

**Step 3: Use virtual key (not master key):**
```python
import anthropic

client = anthropic.Anthropic(
    api_key="sk-litellm-abc123...",  # Virtual key, NOT master key
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## How Authentication Works

### Header Mapping

The Anthropic SDK automatically converts the `api_key` parameter to the `x-api-key` header:

```python
# Your code
client = anthropic.Anthropic(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

# What gets sent to proxy
# Headers:
#   x-api-key: sk-1234
#   anthropic-version: 2023-06-01
```

LiteLLM proxy checks the `x-api-key` header for authentication.

### Authentication Flow

```
1. Your Code
   client = anthropic.Anthropic(api_key="sk-1234", ...)
   
2. Anthropic SDK adds header
   x-api-key: sk-1234
   
3. LiteLLM Proxy receives request
   Checks x-api-key header
   
4. Proxy validates key
   - If master_key set: Must match master key or be valid virtual key
   - If no master_key: Any key accepted
   
5. If valid: Process request
   If invalid: Return 401 Unauthorized
```

---

## Different Authentication Methods

### Method 1: Direct Master Key

**Best for:** Development, testing, single user

```python
import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_MASTER_KEY"],
    base_url="http://localhost:4000"
)
```

**Pros:**
- Simple setup
- No database required

**Cons:**
- No per-user tracking
- No budget limits
- Everyone uses same key

---

### Method 2: Virtual Keys

**Best for:** Production, multi-user, cost control

```python
import anthropic

# Each user/team gets their own key
client = anthropic.Anthropic(
    api_key="sk-litellm-user123",  # Virtual key
    base_url="http://localhost:4000"
)
```

**Pros:**
- Per-user/team tracking
- Budget limits per key
- Rate limiting per key
- Can revoke individual keys

**Cons:**
- Requires database
- More setup

---

### Method 3: Custom Authentication

**Best for:** Integration with existing auth systems

```yaml
# config.yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  custom_auth: custom_auth.py
```

```python
# custom_auth.py
async def user_api_key_auth(request, api_key):
    # Your custom auth logic
    # Check against your database, OAuth, etc.
    if valid_key(api_key):
        return {"user_id": "user123", "team_id": "team456"}
    else:
        raise Exception("Invalid key")
```

---

## Complete Examples

### Example 1: Simple Development Setup

```bash
# Terminal 1: Start proxy (no auth)
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --model anthropic/claude-3-5-sonnet-20241022
```

```python
# Terminal 2: Use with any key
import anthropic

client = anthropic.Anthropic(
    api_key="test-key",  # Any value works
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

---

### Example 2: Production with Master Key

```bash
# Terminal 1: Start proxy with master key
export LITELLM_MASTER_KEY="sk-master-secret-123"
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --model anthropic/claude-3-5-sonnet-20241022
```

```python
# Terminal 2: Use with master key
import anthropic
import os

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_MASTER_KEY"],  # Must match
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)
```

---

### Example 3: Production with Virtual Keys

```bash
# Terminal 1: Start proxy with database
export LITELLM_MASTER_KEY="sk-master-secret-123"
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgresql://user:pass@localhost/litellm"

cat > config.yaml << EOF
model_list:
  - model_name: claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
EOF

litellm --config config.yaml
```

```bash
# Terminal 2: Generate virtual key (using master key)
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-secret-123" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["claude"],
    "max_budget": 100,
    "duration": "30d",
    "metadata": {
      "user": "john@example.com",
      "team": "engineering"
    }
  }'

# Response: {"key": "sk-litellm-xyz789..."}
```

```python
# Terminal 3: Use virtual key (NOT master key)
import anthropic

client = anthropic.Anthropic(
    api_key="sk-litellm-xyz789...",  # Virtual key from above
    base_url="http://localhost:4000"
)

message = client.messages.create(
    model="claude",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(message.content[0].text)

# Proxy tracks:
# - Cost for this request
# - Usage against $100 budget
# - Metadata (user, team)
```

---

## Environment Variables

### Setting Master Key

**Option 1: Export in shell**
```bash
export LITELLM_MASTER_KEY="sk-1234"
litellm --model anthropic/claude-3-5-sonnet-20241022
```

**Option 2: Inline**
```bash
LITELLM_MASTER_KEY="sk-1234" litellm --model anthropic/claude-3-5-sonnet-20241022
```

**Option 3: .env file**
```bash
# .env
LITELLM_MASTER_KEY=sk-1234
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
# Load and start
source .env
litellm --model anthropic/claude-3-5-sonnet-20241022
```

**Option 4: Config file**
```yaml
# config.yaml
general_settings:
  master_key: sk-1234  # Hardcoded (not recommended)
  # OR
  master_key: os.environ/LITELLM_MASTER_KEY  # From environment (recommended)
```

---

## Troubleshooting

### Error: 401 Unauthorized

**Problem:**
```
AuthenticationError: 401 Unauthorized
```

**Solutions:**

1. **Check if master key is set:**
```bash
# Check proxy logs for:
# "Master Key set: True" or "Master Key set: False"
```

2. **If master key is set, ensure you're passing it:**
```python
# Wrong
client = anthropic.Anthropic(
    api_key="wrong-key",
    base_url="http://localhost:4000"
)

# Right
client = anthropic.Anthropic(
    api_key="sk-1234",  # Must match LITELLM_MASTER_KEY
    base_url="http://localhost:4000"
)
```

3. **Check environment variable:**
```bash
echo $LITELLM_MASTER_KEY
# Should output: sk-1234
```

4. **Verify proxy received the key:**
```bash
# Check proxy logs for:
# "Received x-api-key: sk-****"
```

---

### Error: Missing x-api-key header

**Problem:**
```
Missing x-api-key header
```

**Solution:**
Always pass `api_key` to Anthropic client:
```python
# Wrong
client = anthropic.Anthropic(
    base_url="http://localhost:4000"
)

# Right
client = anthropic.Anthropic(
    api_key="sk-1234",  # Required
    base_url="http://localhost:4000"
)
```

---

### Error: Invalid virtual key

**Problem:**
```
Invalid key: sk-litellm-xyz...
```

**Solutions:**

1. **Check key exists:**
```bash
curl http://localhost:4000/key/info \
  -H "Authorization: Bearer sk-master-1234" \
  -H "Content-Type: application/json" \
  -d '{"key": "sk-litellm-xyz..."}'
```

2. **Check key hasn't expired:**
```bash
# Keys have expiration dates
# Check the "expires" field
```

3. **Regenerate key if needed:**
```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-1234" \
  -H "Content-Type: application/json" \
  -d '{"models": ["claude"]}'
```

---

## Security Best Practices

### 1. Never Hardcode Keys

**Bad:**
```python
client = anthropic.Anthropic(
    api_key="sk-1234",  # Hardcoded!
    base_url="http://localhost:4000"
)
```

**Good:**
```python
import os

client = anthropic.Anthropic(
    api_key=os.environ["LITELLM_MASTER_KEY"],
    base_url="http://localhost:4000"
)
```

### 2. Use Virtual Keys in Production

**Bad:** Everyone uses master key
```python
# All users share master key
api_key = "sk-master-1234"
```

**Good:** Each user gets virtual key
```python
# Each user has their own key
api_key = user.get_litellm_key()  # sk-litellm-user123
```

### 3. Set Budget Limits

```bash
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-1234" \
  -d '{
    "models": ["claude"],
    "max_budget": 100,  # $100 limit
    "duration": "30d"
  }'
```

### 4. Rotate Keys Regularly

```bash
# Revoke old key
curl -X POST http://localhost:4000/key/delete \
  -H "Authorization: Bearer sk-master-1234" \
  -d '{"key": "sk-litellm-old..."}'

# Generate new key
curl -X POST http://localhost:4000/key/generate \
  -H "Authorization: Bearer sk-master-1234" \
  -d '{"models": ["claude"]}'
```

### 5. Use HTTPS in Production

```python
# Development
base_url="http://localhost:4000"

# Production
base_url="https://litellm.yourcompany.com"
```

---

## Summary

### Key Points

1. **Pass master key as `api_key` parameter:**
   ```python
   client = anthropic.Anthropic(
       api_key="sk-1234",  # LITELLM_MASTER_KEY
       base_url="http://localhost:4000"
   )
   ```

2. **Anthropic SDK converts to `x-api-key` header automatically**

3. **Three authentication modes:**
   - No auth (development)
   - Master key (simple production)
   - Virtual keys (advanced production)

4. **For production, use virtual keys with budgets and rate limits**

5. **Never hardcode keys - use environment variables**

### Quick Reference

| Scenario | Start Proxy | Use in Code |
|----------|-------------|-------------|
| Development | `litellm --model ...` | `api_key="anything"` |
| Master Key | `LITELLM_MASTER_KEY=sk-1234 litellm ...` | `api_key="sk-1234"` |
| Virtual Keys | `litellm --config config.yaml` | `api_key="sk-litellm-..."` |

---

## Additional Resources

- [LiteLLM Authentication Docs](https://docs.litellm.ai/docs/proxy/virtual_keys)
- [Anthropic SDK Docs](https://docs.anthropic.com/claude/reference/client-sdks)
- [Example Code](./complete_anthropic_proxy_example.py)
