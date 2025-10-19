"""
Reverse Translation: Native Anthropic Format → LiteLLM Proxy

This demonstrates how LiteLLM Proxy can accept requests in NATIVE Anthropic format
and route them to any provider (OpenAI, Azure, etc.) by translating the format.

This is useful when you have existing code using Anthropic's SDK but want to:
1. Route to different providers (OpenAI, Azure, etc.)
2. Add load balancing, fallbacks, cost tracking
3. Maintain Anthropic's API format without code changes
"""

import os

# ============================================================================
# SCENARIO 1: Native Anthropic Format → LiteLLM Proxy → Anthropic
# ============================================================================

print("="*80)
print("SCENARIO 1: Anthropic Format → Proxy → Anthropic (Pass-through)")
print("="*80)

def anthropic_to_anthropic():
    """
    Send Anthropic-format request to proxy, which forwards to Anthropic
    
    This is useful for adding proxy features (auth, logging, rate limiting)
    without changing your Anthropic code.
    """
    import anthropic
    
    # Point Anthropic SDK to LiteLLM proxy instead of Anthropic API
    client = anthropic.Anthropic(
        api_key="sk-1234",  # Your LiteLLM proxy key
        base_url="http://localhost:4000/anthropic"  # Proxy's Anthropic endpoint
    )
    
    # Use native Anthropic format - no changes needed!
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello from Anthropic format!"}
        ]
    )
    
    print(f"Response: {message.content[0].text}")
    print(f"Model: {message.model}")
    print(f"Tokens: {message.usage.input_tokens} in, {message.usage.output_tokens} out")


# ============================================================================
# SCENARIO 2: Native Anthropic Format → LiteLLM Proxy → OpenAI
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 2: Anthropic Format → Proxy → OpenAI (Translation)")
print("="*80)

def anthropic_format_to_openai():
    """
    Send Anthropic-format request to proxy, which translates and sends to OpenAI
    
    This allows you to:
    - Keep your Anthropic SDK code unchanged
    - Route to OpenAI (or any other provider) behind the scenes
    - Switch providers without code changes
    """
    import anthropic
    
    # Same Anthropic SDK code, but proxy routes to OpenAI!
    client = anthropic.Anthropic(
        api_key="sk-1234",
        base_url="http://localhost:4000/anthropic"
    )
    
    # Native Anthropic format
    message = client.messages.create(
        model="gpt-4",  # Proxy translates Anthropic format → OpenAI
        max_tokens=1024,
        system="You are a helpful assistant",
        messages=[
            {"role": "user", "content": "Hello! What model are you?"}
        ]
    )
    
    print(f"Response: {message.content[0].text}")
    print(f"Model used: {message.model}")


# ============================================================================
# SCENARIO 3: How the Translation Works
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 3: Understanding the Translation")
print("="*80)

def show_translation_flow():
    """
    Demonstrates what happens during translation
    """
    
    print("\n1. YOUR CODE (Anthropic SDK):")
    print("-" * 40)
    print("""
    client.messages.create(
        model="gpt-4",
        max_tokens=1024,
        system="You are helpful",
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    """)
    
    print("\n2. WHAT PROXY RECEIVES (Anthropic Format):")
    print("-" * 40)
    print("""
    POST http://localhost:4000/anthropic/v1/messages
    {
        "model": "gpt-4",
        "max_tokens": 1024,
        "system": "You are helpful",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    """)
    
    print("\n3. PROXY TRANSLATES TO (OpenAI Format):")
    print("-" * 40)
    print("""
    POST https://api.openai.com/v1/chat/completions
    {
        "model": "gpt-4",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
    }
    """)
    
    print("\n4. OPENAI RESPONDS:")
    print("-" * 40)
    print("""
    {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Hello! I'm GPT-4..."
            }
        }],
        "usage": {...}
    }
    """)
    
    print("\n5. PROXY TRANSLATES BACK TO (Anthropic Format):")
    print("-" * 40)
    print("""
    {
        "id": "msg-123",
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": "Hello! I'm GPT-4..."
        }],
        "model": "gpt-4",
        "usage": {...}
    }
    """)
    
    print("\n6. YOUR CODE RECEIVES (Anthropic Format):")
    print("-" * 40)
    print("""
    message.content[0].text
    # "Hello! I'm GPT-4..."
    """)


# ============================================================================
# SCENARIO 4: Proxy Configuration for Reverse Translation
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 4: Proxy Configuration")
print("="*80)

def show_proxy_config():
    """
    Show how to configure the proxy for reverse translation
    """
    
    config = """
# config.yaml for LiteLLM Proxy

model_list:
  # Route Anthropic format requests to OpenAI
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
  
  # Route Anthropic format requests to Azure
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4-deployment
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
  
  # Route Anthropic format requests to actual Anthropic
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

# Enable Anthropic endpoint
litellm_settings:
  enable_anthropic_endpoint: true
"""
    
    print("\nProxy Configuration:")
    print("-" * 40)
    print(config)
    
    print("\nStart the proxy:")
    print("-" * 40)
    print("litellm --config config.yaml")
    
    print("\nNow your Anthropic SDK code can route to ANY provider!")


# ============================================================================
# SCENARIO 5: Real-World Use Case - Migration
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 5: Real-World Migration Example")
print("="*80)

def migration_example():
    """
    Real-world scenario: Migrating from Anthropic to OpenAI without code changes
    """
    
    print("\nBEFORE (Direct to Anthropic):")
    print("-" * 40)
    print("""
    import anthropic
    
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    
    # Your production code
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[...]
    )
    """)
    
    print("\nAFTER (Via Proxy to OpenAI):")
    print("-" * 40)
    print("""
    import anthropic
    
    # ONLY CHANGE: Point to proxy
    client = anthropic.Anthropic(
        api_key="sk-proxy-key",
        base_url="http://localhost:4000/anthropic"
    )
    
    # SAME CODE - but now routes to OpenAI!
    response = client.messages.create(
        model="gpt-4",  # Changed model name
        max_tokens=1024,
        messages=[...]  # Same format!
    )
    """)
    
    print("\nBenefits:")
    print("-" * 40)
    print("""
    ✓ No code changes (except 2 lines)
    ✓ Keep Anthropic SDK and format
    ✓ Route to any provider (OpenAI, Azure, etc.)
    ✓ Add fallbacks, load balancing
    ✓ Cost tracking and rate limiting
    ✓ Easy to switch back or A/B test
    """)


# ============================================================================
# SCENARIO 6: Advanced - Tool Calling Translation
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 6: Tool Calling Translation")
print("="*80)

def tool_calling_translation():
    """
    Show how tool calling is translated between formats
    """
    import anthropic
    
    client = anthropic.Anthropic(
        api_key="sk-1234",
        base_url="http://localhost:4000/anthropic"
    )
    
    # Anthropic tool format
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    ]
    
    print("\nYour code (Anthropic format):")
    print("-" * 40)
    print(f"tools = {tools}")
    
    print("\nProxy translates to OpenAI format:")
    print("-" * 40)
    print("""
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
    """)
    
    # Make the call
    message = client.messages.create(
        model="gpt-4",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    )
    
    print("\nResponse (back in Anthropic format):")
    print("-" * 40)
    print(f"Content: {message.content}")


# ============================================================================
# SCENARIO 7: Code Comparison
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 7: Side-by-Side Comparison")
print("="*80)

def code_comparison():
    """
    Side-by-side comparison of different approaches
    """
    
    comparison = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                    APPROACH COMPARISON                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  APPROACH 1: Direct Anthropic SDK                                        ║
║  ────────────────────────────────────────────────────────────────────    ║
║  import anthropic                                                         ║
║  client = anthropic.Anthropic(api_key="sk-ant-...")                      ║
║  response = client.messages.create(...)                                  ║
║                                                                           ║
║  ✓ Native Anthropic features                                             ║
║  ✗ Locked to Anthropic                                                   ║
║  ✗ No fallbacks or load balancing                                        ║
║                                                                           ║
║  ─────────────────────────────────────────────────────────────────────   ║
║                                                                           ║
║  APPROACH 2: LiteLLM Library (OpenAI format)                             ║
║  ────────────────────────────────────────────────────────────────────    ║
║  from litellm import completion                                          ║
║  response = completion(                                                  ║
║      model="anthropic/claude-3-5-sonnet",                                ║
║      messages=[...]  # OpenAI format                                     ║
║  )                                                                        ║
║                                                                           ║
║  ✓ Multi-provider support                                                ║
║  ✓ Unified OpenAI format                                                 ║
║  ✗ Need to change code from Anthropic SDK                                ║
║                                                                           ║
║  ─────────────────────────────────────────────────────────────────────   ║
║                                                                           ║
║  APPROACH 3: LiteLLM Proxy (Anthropic format) ⭐ BEST OF BOTH            ║
║  ────────────────────────────────────────────────────────────────────    ║
║  import anthropic                                                         ║
║  client = anthropic.Anthropic(                                           ║
║      api_key="sk-proxy",                                                 ║
║      base_url="http://localhost:4000/anthropic"                          ║
║  )                                                                        ║
║  response = client.messages.create(...)  # Same Anthropic format!        ║
║                                                                           ║
║  ✓ Keep Anthropic SDK and format                                         ║
║  ✓ Multi-provider support                                                ║
║  ✓ Fallbacks, load balancing                                             ║
║  ✓ Cost tracking, rate limiting                                          ║
║  ✓ Minimal code changes (2 lines)                                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
    
    print(comparison)


# ============================================================================
# Key Files in LiteLLM for Reverse Translation
# ============================================================================

print("\n" + "="*80)
print("KEY FILES FOR REVERSE TRANSLATION")
print("="*80)

def show_key_files():
    """
    Show which files handle the reverse translation
    """
    
    files = """
LiteLLM Reverse Translation Architecture:

1. Anthropic Endpoint (Proxy)
   litellm/proxy/anthropic_endpoints/endpoints.py
   └─ Accepts native Anthropic /v1/messages format
   └─ Routes to appropriate handler

2. Experimental Pass-Through (Anthropic → Anthropic)
   litellm/llms/anthropic/experimental_pass_through/
   ├─ messages/transformation.py
   │  └─ No transformation needed (pass-through)
   └─ messages/handler.py
      └─ Direct forwarding to Anthropic API

3. Adapter (Anthropic → OpenAI)
   litellm/llms/anthropic/experimental_pass_through/adapters/
   ├─ transformation.py
   │  └─ Converts Anthropic format → OpenAI format
   └─ handler.py
      └─ Routes to OpenAI (or other providers)

4. Response Translation (OpenAI → Anthropic)
   litellm/llms/anthropic/experimental_pass_through/adapters/
   └─ transformation.py
      └─ Converts OpenAI response → Anthropic format

Flow:
┌─────────────────────────────────────────────────────────────┐
│ Your Code (Anthropic SDK)                                   │
│ client.messages.create(...)                                 │
└────────────────────────┬────────────────────────────────────┘
                         │ Anthropic Format
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ LiteLLM Proxy                                               │
│ /anthropic/v1/messages endpoint                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────┴────┐
                    │         │
         Anthropic  │         │  Other Provider
                    ▼         ▼
        ┌──────────────┐  ┌──────────────┐
        │ Pass-through │  │   Adapter    │
        │ (no change)  │  │  (translate) │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               ▼                 ▼
        ┌──────────────┐  ┌──────────────┐
        │  Anthropic   │  │    OpenAI    │
        │     API      │  │     API      │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               │                 │ Translate back
               │                 ▼
               │          ┌──────────────┐
               │          │   Adapter    │
               │          │  (response)  │
               │          └──────┬───────┘
               │                 │
               └────────┬────────┘
                        │ Anthropic Format
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ Your Code (Anthropic SDK)                                   │
│ message.content[0].text                                     │
└─────────────────────────────────────────────────────────────┘
"""
    
    print(files)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("REVERSE TRANSLATION: ANTHROPIC FORMAT → LITELLM PROXY")
    print("="*80)
    
    show_translation_flow()
    show_proxy_config()
    migration_example()
    code_comparison()
    show_key_files()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
LiteLLM Proxy supports BIDIRECTIONAL translation:

1. FORWARD: OpenAI format → Anthropic API
   - Use: completion(model="anthropic/claude-3-5-sonnet", ...)
   - Your code uses OpenAI format
   - LiteLLM translates to Anthropic

2. REVERSE: Anthropic format → Any Provider
   - Use: Anthropic SDK with base_url="http://localhost:4000/anthropic"
   - Your code uses Anthropic format
   - LiteLLM translates to OpenAI/Azure/etc.

This gives you maximum flexibility:
✓ Keep your existing code format
✓ Route to any provider
✓ Add proxy features (fallbacks, tracking, etc.)
✓ Easy migration and A/B testing
""")
    
    print("\n" + "="*80)
    print("To try these examples:")
    print("="*80)
    print("""
1. Start the proxy:
   litellm --config config.yaml

2. Run this script:
   python reverse_translation_example.py

3. Or try the examples individually in your code!
""")
