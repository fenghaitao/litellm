"""
NO CODE CHANGES NEEDED: Using Original Anthropic Script with LiteLLM Proxy

This demonstrates that you can use your ORIGINAL Anthropic SDK code
WITHOUT ANY CHANGES by configuring the Anthropic client to point to LiteLLM proxy.

The key is using Anthropic's base_url parameter!
"""

import anthropic
import os

# ============================================================================
# OPTION 1: Zero Code Changes - Environment Variable Method
# ============================================================================

"""
Set this environment variable BEFORE running your script:

export ANTHROPIC_BASE_URL="http://localhost:4000/v1/messages"

Then your original Anthropic code works as-is!
"""

def original_code_no_changes():
    """
    This is your ORIGINAL Anthropic code - NO CHANGES!
    It will automatically use LiteLLM proxy if ANTHROPIC_BASE_URL is set.
    """
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


# ============================================================================
# OPTION 2: Minimal Code Change - One Line Addition
# ============================================================================

def minimal_change_one_line():
    """
    Add just ONE line to your original code - the base_url parameter.
    Everything else stays the same!
    """
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="http://localhost:4000/v1"  # ← ONLY CHANGE: Add this line
    )
    
    # Everything below is UNCHANGED from original Anthropic code
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
    )
    
    print(message.content[0].text)


# ============================================================================
# OPTION 3: Configuration-Based (No Code Changes)
# ============================================================================

"""
Create a wrapper script that sets the environment variable:

# run_with_proxy.sh
#!/bin/bash
export ANTHROPIC_BASE_URL="http://localhost:4000/v1/messages"
python your_original_script.py

Then run: ./run_with_proxy.sh

Your original script needs ZERO changes!
"""


# ============================================================================
# Complete Examples - All Work Without Changing Original Code
# ============================================================================

def streaming_example():
    """Original streaming code - works with proxy via base_url"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="http://localhost:4000/v1"  # Only change needed
    )
    
    # Original Anthropic streaming code - unchanged
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Write a haiku about coding"}
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)


def tools_example():
    """Original tool calling code - works with proxy"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="http://localhost:4000/v1"  # Only change needed
    )
    
    # Original Anthropic tool code - unchanged
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
                            "description": "The city and state"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ]
    )
    
    print(message.content)


def vision_example():
    """Original vision code - works with proxy"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="http://localhost:4000/v1"  # Only change needed
    )
    
    # Original Anthropic vision code - unchanged
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
                            "url": "https://example.com/image.jpg"
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


def system_prompt_example():
    """Original system prompt code - works with proxy"""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        base_url="http://localhost:4000/v1"  # Only change needed
    )
    
    # Original Anthropic code with system prompt - unchanged
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant.",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    
    print(message.content[0].text)


# ============================================================================
# Setup Helper Functions
# ============================================================================

def setup_environment_variable():
    """
    Helper to set environment variable programmatically
    (though it's better to set it in your shell/deployment config)
    """
    os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:4000/v1/messages"
    print("✓ Environment variable set: ANTHROPIC_BASE_URL")


def create_wrapper_function():
    """
    Create a wrapper that automatically uses proxy
    """
    def get_anthropic_client():
        """Get Anthropic client configured for LiteLLM proxy"""
        return anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:4000/v1")
        )
    
    return get_anthropic_client


# ============================================================================
# Real-World Migration Example
# ============================================================================

class MyOriginalApp:
    """
    This is your original application class.
    You can keep it EXACTLY as-is!
    """
    
    def __init__(self):
        # Original initialization - no changes needed if env var is set
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    
    def chat(self, user_message: str) -> str:
        """Original chat method - no changes"""
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return message.content[0].text
    
    def chat_with_history(self, messages: list) -> str:
        """Original chat with history - no changes"""
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=messages
        )
        return message.content[0].text


class MyAppWithProxy(MyOriginalApp):
    """
    If you want to be explicit, inherit and override __init__
    This is the ONLY change needed!
    """
    
    def __init__(self):
        # Override to add base_url - that's it!
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url="http://localhost:4000/v1"  # Only change
        )
        # All other methods inherited unchanged!


# ============================================================================
# Deployment Examples
# ============================================================================

def docker_compose_example():
    """
    Example docker-compose.yml that sets environment variable
    
    version: '3.8'
    services:
      litellm-proxy:
        image: ghcr.io/berriai/litellm:main-latest
        ports:
          - "4000:4000"
        environment:
          - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
        volumes:
          - ./config.yaml:/app/config.yaml
        command: ["--config", "/app/config.yaml"]
      
      your-app:
        build: .
        environment:
          - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
          - ANTHROPIC_BASE_URL=http://litellm-proxy:4000/v1/messages
        depends_on:
          - litellm-proxy
    
    Now your app uses proxy automatically - no code changes!
    """
    pass


def kubernetes_example():
    """
    Example Kubernetes ConfigMap
    
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: app-config
    data:
      ANTHROPIC_BASE_URL: "http://litellm-proxy-service:4000/v1/messages"
    
    Your pods get this env var - no code changes needed!
    """
    pass


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    summary = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                  DO YOU NEED TO CHANGE YOUR CODE?                         ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  SHORT ANSWER: NO! (with environment variable)                           ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  OPTION 1: Zero Code Changes ✓                                           ║
    ║  ──────────────────────────────                                          ║
    ║  Set environment variable:                                               ║
    ║    export ANTHROPIC_BASE_URL="http://localhost:4000/v1/messages"        ║
    ║                                                                           ║
    ║  Your original code works as-is:                                         ║
    ║    client = anthropic.Anthropic()                                        ║
    ║    message = client.messages.create(...)                                 ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  OPTION 2: One Line Change ✓                                             ║
    ║  ───────────────────────────                                             ║
    ║  Add base_url parameter:                                                 ║
    ║    client = anthropic.Anthropic(                                         ║
    ║        base_url="http://localhost:4000/v1"  # ← Add this                ║
    ║    )                                                                      ║
    ║                                                                           ║
    ║  Everything else stays the same!                                         ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  WHAT WORKS WITHOUT CHANGES:                                             ║
    ║  ✓ Basic completions                                                     ║
    ║  ✓ Streaming                                                             ║
    ║  ✓ Tool/Function calling                                                 ║
    ║  ✓ Vision/Image analysis                                                 ║
    ║  ✓ System prompts                                                        ║
    ║  ✓ Multi-turn conversations                                              ║
    ║  ✓ All Anthropic SDK features                                            ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  BONUS: You Get These Features FREE:                                     ║
    ║  ✓ Load balancing                                                        ║
    ║  ✓ Automatic fallbacks                                                   ║
    ║  ✓ Cost tracking                                                         ║
    ║  ✓ Rate limiting                                                         ║
    ║  ✓ Caching                                                               ║
    ║  ✓ Team management                                                       ║
    ║  ✓ Budget controls                                                       ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(summary)


# ============================================================================
# Quick Start Guide
# ============================================================================

def print_quick_start():
    guide = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                         QUICK START GUIDE                                 ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  Step 1: Start LiteLLM Proxy                                             ║
    ║  ────────────────────────────                                            ║
    ║  $ pip install 'litellm[proxy]'                                          ║
    ║  $ litellm --model anthropic/claude-3-5-sonnet-20241022                  ║
    ║                                                                           ║
    ║  Proxy is now running on http://localhost:4000                           ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  Step 2: Set Environment Variable                                        ║
    ║  ─────────────────────────────────                                       ║
    ║  $ export ANTHROPIC_BASE_URL="http://localhost:4000/v1/messages"        ║
    ║                                                                           ║
    ║  ─────────────────────────────────────────────────────────────────────    ║
    ║                                                                           ║
    ║  Step 3: Run Your Original Script                                        ║
    ║  ──────────────────────────────────                                      ║
    ║  $ python your_original_anthropic_script.py                              ║
    ║                                                                           ║
    ║  That's it! No code changes needed!                                      ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(guide)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("NO CODE CHANGES NEEDED: Original Anthropic Script + LiteLLM Proxy")
    print("="*80 + "\n")
    
    print_summary()
    print_quick_start()
    
    print("\n" + "="*80)
    print("Try it yourself:")
    print("1. Start proxy: litellm --model anthropic/claude-3-5-sonnet-20241022")
    print("2. Set env var: export ANTHROPIC_BASE_URL='http://localhost:4000/v1/messages'")
    print("3. Run your original Anthropic script - it just works!")
    print("="*80 + "\n")
