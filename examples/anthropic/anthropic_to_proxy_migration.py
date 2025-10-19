"""
Migration Guide: Original Anthropic SDK → LiteLLM Proxy

This file shows side-by-side comparisons of using the native Anthropic SDK
versus using LiteLLM Proxy with the OpenAI SDK.
"""

# ============================================================================
# BEFORE: Using Native Anthropic SDK
# ============================================================================

"""
# Install: pip install anthropic
"""

import anthropic
import os

# Original Anthropic code
def original_anthropic_basic():
    """Original way: Using Anthropic SDK directly"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
    )
    
    print(message.content[0].text)


def original_anthropic_with_system():
    """Original: With system prompt"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant specialized in Python programming.",
        messages=[
            {"role": "user", "content": "Explain list comprehensions"}
        ]
    )
    
    print(message.content[0].text)


def original_anthropic_streaming():
    """Original: Streaming response"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Write a haiku about coding"}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)


def original_anthropic_tools():
    """Original: Tool/Function calling"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        messages=[
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    )
    
    print(message.content)


def original_anthropic_vision():
    """Original: Vision/Image analysis"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image"
                    }
                ]
            }
        ]
    )
    
    print(message.content[0].text)


# ============================================================================
# AFTER: Using LiteLLM Proxy with OpenAI SDK
# ============================================================================

"""
# Install: pip install openai
# Start proxy: litellm --config config.yaml
"""

import openai

# Configure OpenAI client to point to LiteLLM proxy
def setup_proxy_client():
    """Setup OpenAI client to use LiteLLM proxy"""
    client = openai.OpenAI(
        api_key="sk-1234",  # Your LiteLLM proxy key (or "anything" for testing)
        base_url="http://localhost:4000"  # LiteLLM proxy URL
    )
    return client


def proxy_basic():
    """After: Basic completion via proxy"""
    client = setup_proxy_client()
    
    response = client.chat.completions.create(
        model="claude",  # Model name from your config
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
    )
    
    print(response.choices[0].message.content)


def proxy_with_system():
    """After: With system prompt via proxy"""
    client = setup_proxy_client()
    
    response = client.chat.completions.create(
        model="claude",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant specialized in Python programming."
            },
            {
                "role": "user",
                "content": "Explain list comprehensions"
            }
        ]
    )
    
    print(response.choices[0].message.content)


def proxy_streaming():
    """After: Streaming via proxy"""
    client = setup_proxy_client()
    
    stream = client.chat.completions.create(
        model="claude",
        messages=[
            {"role": "user", "content": "Write a haiku about coding"}
        ],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)


def proxy_tools():
    """After: Tool/Function calling via proxy"""
    client = setup_proxy_client()
    
    response = client.chat.completions.create(
        model="claude",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The unit of temperature"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        messages=[
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    )
    
    print(response.choices[0].message.tool_calls)


def proxy_vision():
    """After: Vision/Image analysis via proxy"""
    client = setup_proxy_client()
    
    response = client.chat.completions.create(
        model="claude",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ]
    )
    
    print(response.choices[0].message.content)


# ============================================================================
# Advanced Examples
# ============================================================================

def proxy_with_metadata():
    """After: With metadata and user tracking"""
    client = setup_proxy_client()
    
    response = client.chat.completions.create(
        model="claude",
        messages=[
            {"role": "user", "content": "Hello!"}
        ],
        # LiteLLM proxy specific: track user/team
        extra_body={
            "metadata": {
                "user_id": "user-123",
                "team_id": "team-456",
                "tags": ["production", "customer-support"]
            }
        }
    )
    
    print(response.choices[0].message.content)


def proxy_with_fallbacks():
    """After: Automatic fallbacks configured in proxy"""
    client = setup_proxy_client()
    
    # The proxy handles fallbacks automatically based on config
    # If claude-sonnet fails, it can fallback to claude-haiku
    response = client.chat.completions.create(
        model="claude",  # Proxy handles fallback logic
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(response.choices[0].message.content)
    print(f"Model used: {response.model}")  # Shows which model actually responded


def proxy_async_example():
    """After: Async requests via proxy"""
    import asyncio
    
    async def make_request():
        client = openai.AsyncOpenAI(
            api_key="sk-1234",
            base_url="http://localhost:4000"
        )
        
        response = await client.chat.completions.create(
            model="claude",
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
        )
        
        return response.choices[0].message.content
    
    result = asyncio.run(make_request())
    print(result)


# ============================================================================
# Comparison Summary
# ============================================================================

def print_comparison():
    """Print side-by-side comparison"""
    
    comparison = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                    ANTHROPIC SDK vs LITELLM PROXY                         ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  ANTHROPIC SDK                    │  LITELLM PROXY                       ║
    ║  ─────────────────────────────────┼──────────────────────────────────    ║
    ║                                   │                                       ║
    ║  import anthropic                 │  import openai                       ║
    ║                                   │                                       ║
    ║  client = anthropic.Anthropic()   │  client = openai.OpenAI(             ║
    ║                                   │      base_url="http://localhost:4000"║
    ║                                   │  )                                    ║
    ║                                   │                                       ║
    ║  client.messages.create(...)      │  client.chat.completions.create(...) ║
    ║                                   │                                       ║
    ║  ─────────────────────────────────┼──────────────────────────────────    ║
    ║                                   │                                       ║
    ║  PROS:                            │  PROS:                               ║
    ║  • Native Anthropic features      │  • OpenAI-compatible (standard)      ║
    ║  • Direct API access              │  • Multi-provider support            ║
    ║  • Latest features first          │  • Built-in fallbacks                ║
    ║                                   │  • Cost tracking                     ║
    ║                                   │  • Rate limiting                     ║
    ║                                   │  • Caching                           ║
    ║                                   │  • Load balancing                    ║
    ║                                   │  • Team management                   ║
    ║                                   │  • Unified interface                 ║
    ║                                   │                                       ║
    ║  CONS:                            │  CONS:                               ║
    ║  • Anthropic-specific code        │  • Extra infrastructure              ║
    ║  • No fallbacks                   │  • Slight latency overhead           ║
    ║  • Manual cost tracking           │  • Feature lag (new features)        ║
    ║  • No rate limiting               │                                       ║
    ║  • Vendor lock-in                 │                                       ║
    ║                                   │                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    
    print(comparison)


# ============================================================================
# Migration Checklist
# ============================================================================

def migration_checklist():
    """Print migration checklist"""
    
    checklist = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         MIGRATION CHECKLIST                               ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  □ 1. Install LiteLLM                                                    ║
    ║       pip install 'litellm[proxy]'                                       ║
    ║                                                                           ║
    ║  □ 2. Create config.yaml                                                 ║
    ║       See example below                                                  ║
    ║                                                                           ║
    ║  □ 3. Start LiteLLM proxy                                                ║
    ║       litellm --config config.yaml                                       ║
    ║                                                                           ║
    ║  □ 4. Update your code                                                   ║
    ║       - Replace: import anthropic                                        ║
    ║       - With: import openai                                              ║
    ║                                                                           ║
    ║  □ 5. Update client initialization                                       ║
    ║       - Replace: anthropic.Anthropic()                                   ║
    ║       - With: openai.OpenAI(base_url="http://localhost:4000")           ║
    ║                                                                           ║
    ║  □ 6. Update method calls                                                ║
    ║       - Replace: client.messages.create()                                ║
    ║       - With: client.chat.completions.create()                           ║
    ║                                                                           ║
    ║  □ 7. Update message format                                              ║
    ║       - system parameter → system role message                           ║
    ║       - tools input_schema → tools parameters                            ║
    ║                                                                           ║
    ║  □ 8. Test thoroughly                                                    ║
    ║       - Basic completions                                                ║
    ║       - Streaming                                                        ║
    ║       - Tool calling                                                     ║
    ║       - Vision (if used)                                                 ║
    ║                                                                           ║
    ║  □ 9. Monitor and optimize                                               ║
    ║       - Check proxy logs                                                 ║
    ║       - Monitor costs                                                    ║
    ║       - Configure fallbacks                                              ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    
    print(checklist)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANTHROPIC SDK → LITELLM PROXY MIGRATION GUIDE")
    print("="*80)
    
    print_comparison()
    migration_checklist()
    
    print("\n" + "="*80)
    print("See the examples above for detailed code comparisons")
    print("="*80 + "\n")
