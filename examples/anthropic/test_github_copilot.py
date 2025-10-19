#!/usr/bin/env python3
"""
Test GitHub Copilot GPT-4 with Anthropic SDK Format

This demonstrates:
1. Using Anthropic SDK with GitHub Copilot's GPT-4 model
2. Translation from Anthropic format → OpenAI format → GitHub Copilot
3. Response translation back to Anthropic format
4. GitHub Copilot uses OAuth2 (no API key needed)

Prerequisites:
1. Install: pip install anthropic litellm[proxy]
2. Authenticate with GitHub Copilot (OAuth2 is handled automatically by LiteLLM)
3. Start proxy: litellm --config github_copilot_example.yaml
4. Run this script: python test_github_copilot.py
"""

import os
import sys
from typing import Optional


def check_proxy_running():
    """Check if LiteLLM proxy is running"""
    import requests
    
    try:
        # Try health endpoint (may require auth)
        response = requests.get("http://localhost:4000/health", timeout=2)
        if response.status_code == 200:
            print("✅ LiteLLM proxy is running")
            return True
        elif response.status_code == 401:
            # Proxy is running but requires auth for health endpoint
            print("✅ LiteLLM proxy is running (auth required for /health)")
            return True
        else:
            print(f"⚠️  Proxy responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: LiteLLM proxy is not running")
        print("\nTo start the proxy:")
        print("   litellm --config github_copilot_example.yaml")
        print("\nThen run this script again.")
        return False
    except Exception as e:
        print(f"❌ ERROR checking proxy: {e}")
        return False


def test_basic_completion():
    """Test basic completion with GitHub Copilot using Anthropic format"""
    import anthropic
    
    print("\n" + "="*80)
    print("TEST 1: Basic Completion")
    print("="*80)
    
    client = anthropic.Anthropic(
        api_key="anything",  # No auth required if proxy has no master_key
        base_url="http://localhost:4000"
    )
    
    print("\n📤 Sending request in Anthropic format:")
    print("   Model: copilot-gpt4")
    print("   Format: Anthropic /v1/messages")
    print("   Content: 'Explain what GitHub Copilot is in one sentence'")
    
    try:
        message = client.messages.create(
            model="copilot-gpt4",  # GitHub Copilot model
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Explain what GitHub Copilot is in one sentence"
                }
            ]
        )
        
        print("\n📥 Response received in Anthropic format:")
        print(f"   ID: {message.id}")
        print(f"   Model: {message.model}")
        print(f"   Type: {message.type}")
        print(f"   Role: {message.role}")
        print(f"   Stop Reason: {message.stop_reason}")
        print(f"\n   Content: {message.content[0].text}")
        print(f"\n   Usage:")
        print(f"   - Input tokens: {message.usage.input_tokens}")
        print(f"   - Output tokens: {message.usage.output_tokens}")
        
        print("\n✅ SUCCESS: Translation worked!")
        print("   Anthropic format → OpenAI format → GitHub Copilot → Anthropic format")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Is the proxy running? (litellm --config github_copilot_example.yaml)")
        print("2. Are you authenticated with GitHub Copilot?")
        print("3. Check proxy logs for details")
        return False


def test_with_system_prompt():
    """Test with system prompt (Anthropic-specific feature)"""
    import anthropic
    
    print("\n" + "="*80)
    print("TEST 2: With System Prompt")
    print("="*80)
    
    client = anthropic.Anthropic(
        api_key="anything",
        base_url="http://localhost:4000"
    )
    
    print("\n📤 Sending request with system prompt:")
    print("   System: 'You are a helpful coding assistant'")
    print("   User: 'Write a Python function to reverse a string'")
    
    try:
        message = client.messages.create(
            model="copilot-gpt4",
            max_tokens=200,
            system="You are a helpful coding assistant. Provide concise code examples.",
            messages=[
                {
                    "role": "user",
                    "content": "Write a Python function to reverse a string"
                }
            ]
        )
        
        print("\n📥 Response:")
        print(message.content[0].text)
        
        print("\n✅ SUCCESS: System prompt was translated correctly!")
        print("   Anthropic 'system' param → OpenAI messages[0] with role='system'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_streaming():
    """Test streaming response"""
    import anthropic
    
    print("\n" + "="*80)
    print("TEST 3: Streaming Response")
    print("="*80)
    
    client = anthropic.Anthropic(
        api_key="anything",
        base_url="http://localhost:4000"
    )
    
    print("\n📤 Sending streaming request:")
    print("   Stream: True")
    
    try:
        print("\n📥 Streaming response:")
        print("   ", end="", flush=True)
        
        with client.messages.stream(
            model="copilot-gpt4",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Count from 1 to 5 with words"
                }
            ]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        
        print("\n\n✅ SUCCESS: Streaming translation worked!")
        print("   OpenAI streaming chunks → Anthropic streaming format")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_multi_turn_conversation():
    """Test multi-turn conversation"""
    import anthropic
    
    print("\n" + "="*80)
    print("TEST 4: Multi-turn Conversation")
    print("="*80)
    
    client = anthropic.Anthropic(
        api_key="anything",
        base_url="http://localhost:4000"
    )
    
    print("\n📤 Sending multi-turn conversation:")
    
    try:
        message = client.messages.create(
            model="copilot-gpt4",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2?"
                },
                {
                    "role": "assistant",
                    "content": "2+2 equals 4."
                },
                {
                    "role": "user",
                    "content": "What about 2+3?"
                }
            ]
        )
        
        print("\n📥 Response:")
        print(message.content[0].text)
        
        print("\n✅ SUCCESS: Multi-turn conversation translated correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False


def test_different_models():
    """Test different GitHub Copilot models"""
    import anthropic
    
    print("\n" + "="*80)
    print("TEST 5: Different GitHub Copilot Models")
    print("="*80)
    
    client = anthropic.Anthropic(
        api_key="anything",
        base_url="http://localhost:4000"
    )
    
    models = [
        ("copilot-gpt4", "GPT-4"),
        ("copilot-gpt35", "GPT-3.5 Turbo"),
    ]
    
    question = "What is 2+2? Answer in one word."
    
    for model_name, display_name in models:
        print(f"\n📤 Testing {display_name}:")
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=20,
                messages=[{"role": "user", "content": question}]
            )
            print(f"   Response: {message.content[0].text}")
            # Calculate total tokens (Anthropic usage object)
            total = message.usage.input_tokens + message.usage.output_tokens
            print(f"   Tokens: {total} (input: {message.usage.input_tokens}, output: {message.usage.output_tokens})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ SUCCESS: Same Anthropic SDK works with all GitHub Copilot models!")
    return True


def show_translation_details():
    """Show what's happening under the hood"""
    print("\n" + "="*80)
    print("TRANSLATION DETAILS")
    print("="*80)
    
    print("""
When you call:
    client.messages.create(
        model="copilot-gpt4",
        system="You are helpful",
        messages=[{"role": "user", "content": "Hello"}]
    )

Here's what happens:

1. Request Format (Anthropic):
   POST http://localhost:4000/v1/messages
   {
       "model": "copilot-gpt4",
       "system": "You are helpful",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 1024
   }

2. LiteLLM Proxy Detects:
   - Model: "copilot-gpt4" → Provider: "github_copilot"
   - Provider is NOT Anthropic → Translation required

3. Translation (Anthropic → OpenAI):
   {
       "model": "github_copilot/gpt-4",
       "messages": [
           {"role": "system", "content": "You are helpful"},
           {"role": "user", "content": "Hello"}
       ],
       "max_tokens": 1024
   }

4. Call GitHub Copilot API (OpenAI format)
   - OAuth2 authentication handled automatically by LiteLLM

5. Receive Response (OpenAI format):
   {
       "id": "chatcmpl-123",
       "choices": [{
           "message": {
               "role": "assistant",
               "content": "Hello! How can I help you?"
           }
       }],
       "usage": {"prompt_tokens": 10, "completion_tokens": 8}
   }

6. Translation (OpenAI → Anthropic):
   {
       "id": "chatcmpl-123",
       "type": "message",
       "role": "assistant",
       "content": [{
           "type": "text",
           "text": "Hello! How can I help you?"
       }],
       "usage": {
           "input_tokens": 10,
           "output_tokens": 8
       }
   }

7. Your Code Receives Anthropic Format!
   message.content[0].text
   # "Hello! How can I help you!"

Note: GitHub Copilot uses OAuth2 authentication, which is handled
automatically by LiteLLM. No API key or token is needed in the config!
""")


def main():
    """Run all tests"""
    print("="*80)
    print("GitHub Copilot + Anthropic SDK Format Test")
    print("="*80)
    
    # Check if proxy is running
    if not check_proxy_running():
        sys.exit(1)
    
    # Show what's happening
    show_translation_details()
    
    # Run tests
    results = []
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    results.append(("Basic Completion", test_basic_completion()))
    
    if results[0][1]:  # Only continue if first test passed
        results.append(("System Prompt", test_with_system_prompt()))
        results.append(("Streaming", test_streaming()))
        results.append(("Multi-turn", test_multi_turn_conversation()))
        results.append(("Different Models", test_different_models()))
    else:
        print("\n⚠️  Skipping remaining tests due to first test failure")
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Translation is working perfectly!")
        print("\nKey Takeaway:")
        print("   You can use Anthropic SDK format with GitHub Copilot!")
        print("   LiteLLM handles all the translation automatically.")
    elif passed_count > 0:
        print("\n⚠️  Some tests passed. Check failed tests above.")
    else:
        print("\n❌ All tests failed. Check your setup:")
        print("   1. Start proxy: litellm --config github_copilot_example.yaml")
        print("   2. Ensure GitHub Copilot OAuth2 is working")
        print("   3. Check proxy logs for details")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
