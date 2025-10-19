"""
Quick Start: LiteLLM with Anthropic Claude

This is a minimal example to get started with LiteLLM and Anthropic's Claude models.
"""

import os
from litellm import completion

# Step 1: Set your API key
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."  # Replace with your actual key

# Step 2: Make a simple call
response = completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude! Introduce yourself."}
    ]
)

# Step 3: Get the response
print(response.choices[0].message.content)

# That's it! LiteLLM handles all the translation to Anthropic's API format.
