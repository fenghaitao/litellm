"""
LiteLLM + Anthropic API Examples

This demonstrates how LiteLLM provides a unified interface to call Anthropic's Claude models
using the OpenAI-compatible format.
"""

import os
from litellm import completion, acompletion
import asyncio

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"


# ============================================================================
# Example 1: Basic Chat Completion with Claude
# ============================================================================
def basic_completion():
    """Simple chat completion with Claude"""
    print("\n=== Example 1: Basic Completion ===")
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",  # or claude-3-opus, claude-3-haiku
        messages=[
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens}")


# ============================================================================
# Example 2: Streaming Response
# ============================================================================
def streaming_completion():
    """Stream responses from Claude"""
    print("\n=== Example 2: Streaming ===")
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "Write a haiku about AI"}
        ],
        stream=True
    )
    
    print("Streaming response: ", end="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


# ============================================================================
# Example 3: Async Completion
# ============================================================================
async def async_completion():
    """Async call to Claude"""
    print("\n=== Example 3: Async Completion ===")
    
    response = await acompletion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "What are the benefits of async programming?"}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")


# ============================================================================
# Example 4: Multi-turn Conversation
# ============================================================================
def conversation():
    """Multi-turn conversation with Claude"""
    print("\n=== Example 4: Multi-turn Conversation ===")
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What's the population?"}
    ]
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=messages
    )
    
    print(f"Response: {response.choices[0].message.content}")


# ============================================================================
# Example 5: System Prompts
# ============================================================================
def system_prompt():
    """Using system prompts with Claude"""
    print("\n=== Example 5: System Prompt ===")
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that speaks like a pirate."
            },
            {
                "role": "user", 
                "content": "Tell me about machine learning"
            }
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")


# ============================================================================
# Example 6: Tool/Function Calling
# ============================================================================
def tool_calling():
    """Function calling with Claude"""
    print("\n=== Example 6: Tool Calling ===")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        tools=tools
    )
    
    # Check if Claude wants to call a function
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Claude wants to call: {tool_call.function.name}")
        print(f"With arguments: {tool_call.function.arguments}")


# ============================================================================
# Example 7: Vision - Image Analysis
# ============================================================================
def vision_analysis():
    """Analyze images with Claude"""
    print("\n=== Example 7: Vision (Image Analysis) ===")
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's in this image?"
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
    
    print(f"Response: {response.choices[0].message.content}")


# ============================================================================
# Example 8: Advanced Parameters
# ============================================================================
def advanced_parameters():
    """Using advanced Anthropic-specific parameters"""
    print("\n=== Example 8: Advanced Parameters ===")
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "Write a creative story"}
        ],
        temperature=0.9,  # Higher temperature for more creativity
        max_tokens=500,   # Limit response length
        top_p=0.95,       # Nucleus sampling
        stop=["\n\n"]     # Stop sequences
    )
    
    print(f"Response: {response.choices[0].message.content}")


# ============================================================================
# Example 9: Prompt Caching (Cost Optimization)
# ============================================================================
def prompt_caching():
    """Use Anthropic's prompt caching to reduce costs"""
    print("\n=== Example 9: Prompt Caching ===")
    
    # Large context that you want to cache
    large_context = """
    [Large document or context that you'll reuse across multiple requests]
    This could be documentation, a knowledge base, or any large text.
    """ * 10
    
    response = completion(
        model="anthropic/claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": large_context,
                        "cache_control": {"type": "ephemeral"}  # Cache this content
                    }
                ]
            },
            {
                "role": "user",
                "content": "Summarize the key points"
            }
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Cache tokens: {response.usage.prompt_tokens_details.get('cached_tokens', 0)}")


# ============================================================================
# Example 10: Extended Thinking (Claude Sonnet 4)
# ============================================================================
def extended_thinking():
    """Use Claude's extended thinking capability for complex reasoning"""
    print("\n=== Example 10: Extended Thinking ===")
    
    response = completion(
        model="anthropic/claude-3-7-sonnet-20250219",  # Sonnet 4 with thinking
        messages=[
            {
                "role": "user",
                "content": "Solve this logic puzzle: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 5000  # Allow up to 5000 tokens for thinking
        }
    )
    
    # Access thinking content if available
    message = response.choices[0].message
    if hasattr(message, 'content') and isinstance(message.content, list):
        for block in message.content:
            if block.get('type') == 'thinking':
                print(f"Thinking process: {block.get('thinking', '')[:200]}...")
    
    print(f"Final answer: {response.choices[0].message.content}")


# ============================================================================
# Example 11: Error Handling
# ============================================================================
def error_handling():
    """Proper error handling with LiteLLM"""
    print("\n=== Example 11: Error Handling ===")
    
    try:
        response = completion(
            model="anthropic/claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=1  # Intentionally too low
        )
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")


# ============================================================================
# Example 12: Comparing Multiple Models
# ============================================================================
async def compare_models():
    """Compare responses from different Claude models"""
    print("\n=== Example 12: Model Comparison ===")
    
    models = [
        "anthropic/claude-3-5-sonnet-20241022",  # Balanced
        "anthropic/claude-3-5-haiku-20241022",   # Fast & cheap
        "anthropic/claude-3-opus-20240229"       # Most capable
    ]
    
    question = "What is the meaning of life?"
    
    tasks = [
        acompletion(
            model=model,
            messages=[{"role": "user", "content": question}]
        )
        for model in models
    ]
    
    responses = await asyncio.gather(*tasks)
    
    for model, response in zip(models, responses):
        print(f"\n{model}:")
        print(f"Response: {response.choices[0].message.content[:100]}...")
        print(f"Tokens: {response.usage.total_tokens}")


# ============================================================================
# Example 13: Using with LiteLLM Router (Load Balancing)
# ============================================================================
def router_example():
    """Use Router for load balancing across multiple deployments"""
    print("\n=== Example 13: Router (Load Balancing) ===")
    
    from litellm import Router
    
    # Configure multiple Claude deployments
    model_list = [
        {
            "model_name": "claude",
            "litellm_params": {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "api_key": os.environ.get("ANTHROPIC_API_KEY")
            }
        },
        {
            "model_name": "claude",
            "litellm_params": {
                "model": "anthropic/claude-3-5-haiku-20241022",
                "api_key": os.environ.get("ANTHROPIC_API_KEY")
            }
        }
    ]
    
    router = Router(
        model_list=model_list,
        routing_strategy="simple-shuffle",  # or "least-busy", "latency-based"
        num_retries=2
    )
    
    # Router automatically picks the best deployment
    response = router.completion(
        model="claude",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Used model: {response.model}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("LiteLLM + Anthropic API Examples")
    print("=" * 80)
    
    # Run synchronous examples
    try:
        basic_completion()
        streaming_completion()
        conversation()
        system_prompt()
        tool_calling()
        vision_analysis()
        advanced_parameters()
        prompt_caching()
        extended_thinking()
        error_handling()
        router_example()
    except Exception as e:
        print(f"Error in sync examples: {e}")
    
    # Run async examples
    try:
        asyncio.run(async_completion())
        asyncio.run(compare_models())
    except Exception as e:
        print(f"Error in async examples: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
