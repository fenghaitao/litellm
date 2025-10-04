#!/usr/bin/env python3
"""
GitHub Copilot Embedding Example

This example demonstrates how to use GitHub Copilot embeddings through litellm.
It shows various use cases including single text embedding, batch processing,
semantic similarity search, and integration with the existing copilot-api client.

Prerequisites:
1. Install litellm: pip install litellm
2. GitHub Copilot subscription (Individual, Business, or Enterprise)
3. Authenticated GitHub account (the script will guide you through OAuth if needed)

Usage:
    python github_copilot_embedding_example.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import List, Tuple
import json

# Add litellm to path (adjust if needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import litellm
    import numpy as np
    print("✅ Successfully imported required packages")
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: pip install litellm numpy")
    sys.exit(1)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


async def basic_embedding_example():
    """Basic example: Single text embedding."""
    print("\n" + "="*60)
    print("🔤 BASIC EMBEDDING EXAMPLE")
    print("="*60)
    
    try:
        # Simple text embedding
        text = "What are templates in DML programming?"
        
        print(f"📝 Text to embed: '{text}'")
        
        start_time = time.time()
        response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=text
        )
        end_time = time.time()
        
        print(f"✅ Embedding generated successfully!")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"📊 Model: {response.model}")
        print(f"📏 Embedding dimensions: {len(response.data[0]['embedding'])}")
        print(f"💰 Token usage: {response.usage.prompt_tokens} prompt tokens")
        print(f"🆔 First few embedding values: {response.data[0]['embedding'][:5]}")
        
        return response.data[0]['embedding']
        
    except Exception as e:
        print(f"❌ Error in basic embedding: {e}")
        return None


async def batch_embedding_example():
    """Batch example: Multiple texts at once."""
    print("\n" + "="*60)
    print("📦 BATCH EMBEDDING EXAMPLE")
    print("="*60)
    
    try:
        # Multiple texts related to DML and Simics
        texts = [
            "How do reset mechanisms work in DML?",
            "What are the differences between methods and parameters in DML?",
            "How are registers and fields modeled in DML?",
            "What is functional modeling in Simics?",
            "How to implement device templates in DML?"
        ]
        
        print(f"📝 Embedding {len(texts)} texts...")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")
        
        start_time = time.time()
        response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=texts
        )
        end_time = time.time()
        
        print(f"\n✅ Batch embedding completed!")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"📊 Model: {response.model}")
        print(f"📏 Number of embeddings: {len(response.data)}")
        print(f"📏 Embedding dimensions: {len(response.data[0]['embedding'])}")
        print(f"💰 Total token usage: {response.usage.total_tokens} tokens")
        print(f"💰 Average tokens per text: {response.usage.prompt_tokens / len(texts):.1f}")
        
        return texts, [item['embedding'] for item in response.data]
        
    except Exception as e:
        print(f"❌ Error in batch embedding: {e}")
        return None, None


async def custom_dimensions_example():
    """Example with custom embedding dimensions."""
    print("\n" + "="*60)
    print("🎛️  CUSTOM DIMENSIONS EXAMPLE")
    print("="*60)
    
    try:
        text = "DML templates provide reusable code blocks for device modeling"
        
        # Test different dimension sizes
        dimension_sizes = [512, 1024, 1536]
        
        for dims in dimension_sizes:
            print(f"\n📏 Testing {dims} dimensions...")
            
            start_time = time.time()
            response = await litellm.aembedding(
                model="github_copilot/text-embedding-3-small",
                input=text,
                dimensions=dims,
                user="embedding_example_user"
            )
            end_time = time.time()
            
            actual_dims = len(response.data[0]['embedding'])
            print(f"   ✅ Requested: {dims}, Got: {actual_dims} dimensions")
            print(f"   ⏱️  Time: {end_time - start_time:.2f}s")
            print(f"   💰 Tokens: {response.usage.prompt_tokens}")
            
            # Show first few values
            embedding_preview = response.data[0]['embedding'][:3]
            print(f"   🔢 Preview: [{embedding_preview[0]:.4f}, {embedding_preview[1]:.4f}, {embedding_preview[2]:.4f}, ...]")
    
    except Exception as e:
        print(f"❌ Error in custom dimensions example: {e}")


async def semantic_similarity_example():
    """Example: Finding semantically similar texts."""
    print("\n" + "="*60)
    print("🔍 SEMANTIC SIMILARITY EXAMPLE")
    print("="*60)
    
    try:
        # Query and candidate texts
        query = "How to create reusable code in DML?"
        
        candidates = [
            "DML templates allow code reuse across device models",
            "Python functions provide code modularity",
            "Register field definitions in device specifications", 
            "Template instantiation with the 'is' keyword in DML",
            "JavaScript modules for web development",
            "Code inheritance using DML template hierarchies",
            "Database schema design principles",
            "DML parameter passing to template instances"
        ]
        
        print(f"🎯 Query: '{query}'")
        print(f"📋 Candidates ({len(candidates)} texts):")
        for i, candidate in enumerate(candidates, 1):
            print(f"   {i}. {candidate}")
        
        # Get embeddings for all texts
        all_texts = [query] + candidates
        
        print(f"\n🔄 Generating embeddings...")
        start_time = time.time()
        response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=all_texts
        )
        end_time = time.time()
        
        # Extract embeddings
        query_embedding = response.data[0]['embedding']
        candidate_embeddings = [item['embedding'] for item in response.data[1:]]
        
        # Calculate similarities
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            similarity = cosine_similarity(query_embedding, candidate_embedding)
            similarities.append((candidates[i], similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Similarity analysis completed!")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"💰 Token usage: {response.usage.total_tokens} tokens")
        
        print(f"\n🏆 TOP 5 MOST SIMILAR TEXTS:")
        for i, (text, similarity) in enumerate(similarities[:5], 1):
            print(f"   {i}. [{similarity:.3f}] {text}")
        
        print(f"\n⬇️  LEAST SIMILAR TEXTS:")
        for i, (text, similarity) in enumerate(similarities[-2:], 1):
            print(f"   {i}. [{similarity:.3f}] {text}")
            
    except Exception as e:
        print(f"❌ Error in semantic similarity example: {e}")


async def comparison_with_original_client():
    """Compare litellm implementation with original copilot-api client."""
    print("\n" + "="*60)
    print("⚖️  COMPARISON WITH ORIGINAL CLIENT")
    print("="*60)
    
    try:
        # Try to import the original client
        original_available = True
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "copilot-api" / "python-port" / "src"))
            from copilot_api.embedding_client import embed_text
            print("✅ Original copilot-api client available")
        except ImportError:
            print("⚠️  Original copilot-api client not available - skipping comparison")
            original_available = False
        
        if not original_available:
            return
        
        test_text = "Testing embedding consistency between implementations"
        print(f"📝 Test text: '{test_text}'")
        
        # Test original client
        print(f"\n🔄 Testing original copilot-api client...")
        start_time = time.time()
        try:
            original_embedding = await embed_text(test_text)
            original_time = time.time() - start_time
            print(f"   ✅ Original: {len(original_embedding)} dimensions, {original_time:.2f}s")
            original_preview = original_embedding[:3]
            print(f"   🔢 Preview: [{original_preview[0]:.4f}, {original_preview[1]:.4f}, {original_preview[2]:.4f}, ...]")
        except Exception as e:
            print(f"   ❌ Original client error: {e}")
            return
        
        # Test litellm implementation
        print(f"\n🔄 Testing litellm implementation...")
        start_time = time.time()
        try:
            litellm_response = await litellm.aembedding(
                model="github_copilot/text-embedding-3-small",
                input=test_text
            )
            litellm_time = time.time() - start_time
            litellm_embedding = litellm_response.data[0]['embedding']
            print(f"   ✅ Litellm: {len(litellm_embedding)} dimensions, {litellm_time:.2f}s")
            litellm_preview = litellm_embedding[:3]
            print(f"   🔢 Preview: [{litellm_preview[0]:.4f}, {litellm_preview[1]:.4f}, {litellm_preview[2]:.4f}, ...]")
            print(f"   💰 Token usage: {litellm_response.usage.prompt_tokens}")
        except Exception as e:
            print(f"   ❌ Litellm implementation error: {e}")
            return
        
        # Compare results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   📏 Dimensions - Original: {len(original_embedding)}, Litellm: {len(litellm_embedding)}")
        print(f"   ⏱️  Performance - Original: {original_time:.2f}s, Litellm: {litellm_time:.2f}s")
        
        if len(original_embedding) == len(litellm_embedding):
            similarity = cosine_similarity(original_embedding, litellm_embedding)
            print(f"   🎯 Embedding similarity: {similarity:.6f}")
            
            if similarity > 0.99:
                print(f"   ✅ Embeddings are highly consistent!")
            elif similarity > 0.95:
                print(f"   ✅ Embeddings are reasonably consistent")
            else:
                print(f"   ⚠️  Embeddings show some differences")
        else:
            print(f"   ⚠️  Different embedding dimensions - cannot compare directly")
            
    except Exception as e:
        print(f"❌ Error in comparison: {e}")


def synchronous_example():
    """Example using synchronous API calls."""
    print("\n" + "="*60)
    print("🔄 SYNCHRONOUS API EXAMPLE")
    print("="*60)
    
    try:
        texts = [
            "Synchronous embedding example",
            "Non-async function call",
            "Traditional blocking API"
        ]
        
        print(f"📝 Embedding {len(texts)} texts synchronously...")
        
        start_time = time.time()
        response = litellm.embedding(
            model="github_copilot/text-embedding-3-small",
            input=texts,
            encoding_format="float"
        )
        end_time = time.time()
        
        print(f"✅ Synchronous embedding completed!")
        print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
        print(f"📊 Model: {response.model}")
        print(f"📏 Embeddings: {len(response.data)}")
        print(f"💰 Token usage: {response.usage.total_tokens}")
        
        # Show encoding format was applied
        print(f"🔢 Embedding type: {type(response.data[0]['embedding'][0])}")
        
    except Exception as e:
        print(f"❌ Error in synchronous example: {e}")


async def error_handling_example():
    """Example demonstrating error handling."""
    print("\n" + "="*60)
    print("⚠️  ERROR HANDLING EXAMPLE")
    print("="*60)
    
    # Test various error scenarios
    error_cases = [
        {
            "name": "Invalid Model",
            "params": {
                "model": "github_copilot/invalid-model",
                "input": "Test text"
            }
        },
        {
            "name": "Empty Input",
            "params": {
                "model": "github_copilot/text-embedding-3-small",
                "input": ""
            }
        }
    ]
    
    for case in error_cases:
        print(f"\n🧪 Testing: {case['name']}")
        try:
            response = await litellm.aembedding(**case['params'])
            print(f"   ⚠️  Unexpected success: {response}")
        except Exception as e:
            print(f"   ✅ Expected error caught: {type(e).__name__}: {e}")


async def main():
    """Run all examples."""
    print("🚀 GitHub Copilot Embedding Examples with litellm")
    print("=" * 60)
    print("This example demonstrates various GitHub Copilot embedding use cases")
    print("through the litellm library.")
    print("\n📋 Examples included:")
    print("1. Basic single text embedding")
    print("2. Batch processing multiple texts")
    print("3. Custom embedding dimensions")
    print("4. Semantic similarity search")
    print("5. Comparison with original client")
    print("6. Synchronous API usage")
    print("7. Error handling scenarios")
    
    try:
        # Run async examples
        await basic_embedding_example()
        await batch_embedding_example()
        await custom_dimensions_example()
        await semantic_similarity_example()
        await comparison_with_original_client()
        await error_handling_example()
        
        # Run sync example
        synchronous_example()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n💡 Key takeaways:")
        print("✅ GitHub Copilot embeddings work seamlessly through litellm")
        print("✅ Supports both sync and async operations")
        print("✅ OpenAI-compatible parameters (dimensions, user, encoding_format)")
        print("✅ Efficient batch processing for multiple texts")
        print("✅ Easy integration with existing litellm workflows")
        print("✅ Comprehensive error handling and authentication")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set litellm to be less verbose for cleaner output
    litellm.set_verbose = False
    
    # Run the examples
    asyncio.run(main())