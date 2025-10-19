"""
Complete Example: Original Anthropic Script → LiteLLM Proxy

This is a real-world example showing a complete application migrated from
native Anthropic SDK to LiteLLM Proxy.
"""

import os
import json
from typing import List, Dict, Any

# ============================================================================
# ORIGINAL: Using Anthropic SDK
# ============================================================================

print("="*80)
print("ORIGINAL VERSION: Using Anthropic SDK")
print("="*80)

def original_chatbot():
    """
    Original chatbot using Anthropic SDK
    
    Requirements:
    - pip install anthropic
    - export ANTHROPIC_API_KEY=sk-ant-...
    """
    import anthropic
    
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Conversation history
    conversation_history = []
    
    # System prompt
    system_prompt = """You are a helpful AI assistant that specializes in Python programming.
    You provide clear, concise answers with code examples when appropriate."""
    
    print("\nChatbot started! (Type 'quit' to exit)")
    print("-" * 80)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            # Call Anthropic API
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=conversation_history
            )
            
            # Extract assistant response
            assistant_message = response.content[0].text
            
            # Add to history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Display response
            print(f"\nClaude: {assistant_message}")
            
            # Show token usage
            print(f"\n[Tokens: {response.usage.input_tokens} in, "
                  f"{response.usage.output_tokens} out]")
            
        except Exception as e:
            print(f"\nError: {e}")
            # Remove the failed user message
            conversation_history.pop()


def original_batch_processing():
    """
    Original: Process multiple prompts in batch
    """
    import anthropic
    import asyncio
    
    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    prompts = [
        "Explain Python decorators",
        "What are Python generators?",
        "How do Python context managers work?"
    ]
    
    async def process_prompt(prompt: str) -> Dict[str, Any]:
        """Process a single prompt"""
        try:
            response = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "prompt": prompt,
                "response": response.content[0].text,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "success": True
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    async def process_all():
        """Process all prompts concurrently"""
        tasks = [process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    print("\nProcessing batch requests...")
    results = asyncio.run(process_all())
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Prompt: {result['prompt']}")
        if result['success']:
            print(f"Response: {result['response'][:100]}...")
            print(f"Tokens: {result['tokens']}")
        else:
            print(f"Error: {result['error']}")


def original_with_tools():
    """
    Original: Using tools/function calling
    """
    import anthropic
    
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )
    
    # Define tools
    tools = [
        {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        },
        {
            "name": "calculate_percentage_change",
            "description": "Calculate percentage change between two values",
            "input_schema": {
                "type": "object",
                "properties": {
                    "old_value": {"type": "number"},
                    "new_value": {"type": "number"}
                },
                "required": ["old_value", "new_value"]
            }
        }
    ]
    
    # Mock tool implementations
    def get_stock_price(ticker: str) -> float:
        """Mock function - in reality, would call a real API"""
        mock_prices = {"AAPL": 178.50, "GOOGL": 142.30, "MSFT": 380.20}
        return mock_prices.get(ticker.upper(), 100.0)
    
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        return ((new_value - old_value) / old_value) * 100
    
    # User query
    user_query = "What's the current price of AAPL stock?"
    
    print(f"\nUser: {user_query}")
    
    # First API call
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_query}]
    )
    
    # Check if Claude wants to use a tool
    if response.stop_reason == "tool_use":
        # Extract tool use
        tool_use = next(block for block in response.content if block.type == "tool_use")
        
        print(f"\nClaude wants to call: {tool_use.name}")
        print(f"With arguments: {tool_use.input}")
        
        # Execute the tool
        if tool_use.name == "get_stock_price":
            result = get_stock_price(tool_use.input["ticker"])
            print(f"Tool result: ${result}")
            
            # Send result back to Claude
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=tools,
                messages=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use.id,
                                "content": str(result)
                            }
                        ]
                    }
                ]
            )
            
            # Get final response
            final_response = response.content[0].text
            print(f"\nClaude: {final_response}")
    else:
        print(f"\nClaude: {response.content[0].text}")


# ============================================================================
# MIGRATED: Using LiteLLM Proxy with OpenAI SDK
# ============================================================================

print("\n" + "="*80)
print("MIGRATED VERSION: Using LiteLLM Proxy")
print("="*80)

def proxy_chatbot():
    """
    Migrated chatbot using LiteLLM Proxy
    
    Requirements:
    - pip install openai
    - Start proxy: litellm --config config.yaml
    - Proxy running at http://localhost:4000
    """
    import openai
    
    client = openai.OpenAI(
        api_key="sk-1234",  # Your LiteLLM proxy key
        base_url="http://localhost:4000"
    )
    
    # Conversation history
    conversation_history = []
    
    # System prompt (now as a message)
    system_message = {
        "role": "system",
        "content": """You are a helpful AI assistant that specializes in Python programming.
        You provide clear, concise answers with code examples when appropriate."""
    }
    
    print("\nChatbot started! (Type 'quit' to exit)")
    print("-" * 80)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Build messages (system + history + new message)
        messages = [system_message] + conversation_history + [
            {"role": "user", "content": user_input}
        ]
        
        try:
            # Call LiteLLM Proxy (which calls Anthropic)
            response = client.chat.completions.create(
                model="claude",  # Model name from config.yaml
                messages=messages,
                max_tokens=1024
            )
            
            # Extract assistant response
            assistant_message = response.choices[0].message.content
            
            # Add to history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Display response
            print(f"\nClaude: {assistant_message}")
            
            # Show token usage
            usage = response.usage
            print(f"\n[Tokens: {usage.prompt_tokens} in, "
                  f"{usage.completion_tokens} out]")
            print(f"[Model: {response.model}]")
            
        except Exception as e:
            print(f"\nError: {e}")


def proxy_batch_processing():
    """
    Migrated: Process multiple prompts in batch
    """
    import openai
    import asyncio
    
    client = openai.AsyncOpenAI(
        api_key="sk-1234",
        base_url="http://localhost:4000"
    )
    
    prompts = [
        "Explain Python decorators",
        "What are Python generators?",
        "How do Python context managers work?"
    ]
    
    async def process_prompt(prompt: str) -> Dict[str, Any]:
        """Process a single prompt"""
        try:
            response = await client.chat.completions.create(
                model="claude",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            return {
                "prompt": prompt,
                "response": response.choices[0].message.content,
                "tokens": response.usage.total_tokens,
                "model": response.model,
                "success": True
            }
        except Exception as e:
            return {
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    async def process_all():
        """Process all prompts concurrently"""
        tasks = [process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    print("\nProcessing batch requests...")
    results = asyncio.run(process_all())
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Prompt: {result['prompt']}")
        if result['success']:
            print(f"Response: {result['response'][:100]}...")
            print(f"Tokens: {result['tokens']}")
            print(f"Model: {result['model']}")
        else:
            print(f"Error: {result['error']}")


def proxy_with_tools():
    """
    Migrated: Using tools/function calling
    """
    import openai
    
    client = openai.OpenAI(
        api_key="sk-1234",
        base_url="http://localhost:4000"
    )
    
    # Define tools (OpenAI format)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"
                        }
                    },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_percentage_change",
                "description": "Calculate percentage change between two values",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "old_value": {"type": "number"},
                        "new_value": {"type": "number"}
                    },
                    "required": ["old_value", "new_value"]
                }
            }
        }
    ]
    
    # Mock tool implementations
    def get_stock_price(ticker: str) -> float:
        """Mock function - in reality, would call a real API"""
        mock_prices = {"AAPL": 178.50, "GOOGL": 142.30, "MSFT": 380.20}
        return mock_prices.get(ticker.upper(), 100.0)
    
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """Calculate percentage change"""
        return ((new_value - old_value) / old_value) * 100
    
    # User query
    user_query = "What's the current price of AAPL stock?"
    
    print(f"\nUser: {user_query}")
    
    # First API call
    response = client.chat.completions.create(
        model="claude",
        messages=[{"role": "user", "content": user_query}],
        tools=tools,
        max_tokens=1024
    )
    
    # Check if Claude wants to use a tool
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        
        print(f"\nClaude wants to call: {tool_call.function.name}")
        print(f"With arguments: {tool_call.function.arguments}")
        
        # Execute the tool
        if tool_call.function.name == "get_stock_price":
            args = json.loads(tool_call.function.arguments)
            result = get_stock_price(args["ticker"])
            print(f"Tool result: ${result}")
            
            # Send result back to Claude
            response = client.chat.completions.create(
                model="claude",
                messages=[
                    {"role": "user", "content": user_query},
                    response.choices[0].message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }
                ],
                tools=tools,
                max_tokens=1024
            )
            
            # Get final response
            final_response = response.choices[0].message.content
            print(f"\nClaude: {final_response}")
    else:
        print(f"\nClaude: {response.choices[0].message.content}")


# ============================================================================
# Additional Proxy Benefits
# ============================================================================

def proxy_benefits_demo():
    """
    Demonstrate additional benefits of using LiteLLM Proxy
    """
    import openai
    
    client = openai.OpenAI(
        api_key="sk-1234",
        base_url="http://localhost:4000"
    )
    
    print("\n" + "="*80)
    print("ADDITIONAL PROXY BENEFITS")
    print("="*80)
    
    # 1. Automatic fallbacks
    print("\n1. Automatic Fallbacks:")
    print("   If claude-sonnet fails, proxy automatically tries claude-haiku")
    print("   (Configured in config.yaml)")
    
    # 2. Cost tracking
    print("\n2. Cost Tracking:")
    print("   Proxy automatically tracks costs per user/team/key")
    print("   View at: http://localhost:4000/spend/logs")
    
    # 3. Rate limiting
    print("\n3. Rate Limiting:")
    print("   Set limits per key: 100 requests/minute")
    print("   Prevents accidental overspending")
    
    # 4. Caching
    print("\n4. Caching:")
    print("   Identical requests are cached (saves cost & latency)")
    
    # 5. Load balancing
    print("\n5. Load Balancing:")
    print("   Distribute requests across multiple API keys")
    print("   Increases throughput and reliability")
    
    # 6. Unified interface
    print("\n6. Unified Interface:")
    print("   Same code works with OpenAI, Anthropic, Azure, etc.")
    print("   Just change model name in config!")
    
    # Example: Using metadata for tracking
    print("\n7. Example: Request with metadata")
    response = client.chat.completions.create(
        model="claude",
        messages=[{"role": "user", "content": "Hello!"}],
        extra_body={
            "metadata": {
                "user_id": "user-123",
                "team_id": "team-456",
                "environment": "production"
            }
        }
    )
    print(f"   Response: {response.choices[0].message.content}")
    print("   Metadata tracked in proxy logs!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPLETE ANTHROPIC → LITELLM PROXY MIGRATION EXAMPLE")
    print("="*80)
    
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS")
    print("="*80)
    print("""
    1. Install dependencies:
       pip install anthropic openai litellm[proxy]
    
    2. Set environment variables:
       export ANTHROPIC_API_KEY=sk-ant-...
    
    3. Start LiteLLM proxy (in another terminal):
       litellm --config proxy_config.yaml
    
    4. Run this script:
       python complete_anthropic_proxy_example.py
    """)
    
    print("\n" + "="*80)
    print("Choose an example to run:")
    print("="*80)
    print("1. Chatbot (Original Anthropic SDK)")
    print("2. Chatbot (LiteLLM Proxy)")
    print("3. Batch Processing (Original)")
    print("4. Batch Processing (Proxy)")
    print("5. Tool Calling (Original)")
    print("6. Tool Calling (Proxy)")
    print("7. Proxy Benefits Demo")
    print("8. Exit")
    
    choice = input("\nEnter choice (1-8): ")
    
    if choice == "1":
        original_chatbot()
    elif choice == "2":
        proxy_chatbot()
    elif choice == "3":
        original_batch_processing()
    elif choice == "4":
        proxy_batch_processing()
    elif choice == "5":
        original_with_tools()
    elif choice == "6":
        proxy_with_tools()
    elif choice == "7":
        proxy_benefits_demo()
    else:
        print("Exiting...")
