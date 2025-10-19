#!/usr/bin/env python3
"""
Simple GitHub Copilot Embedding Example

A quick, practical example showing how to use GitHub Copilot embeddings with litellm.
This example focuses on common use cases with minimal setup.

Prerequisites:
- pip install litellm
- GitHub Copilot subscription
- Authenticated GitHub account

Usage:
    python github_copilot_embedding_simple.py
"""

import asyncio
import litellm
from typing import List


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0.0


async def single_text_embedding():
    """Generate embedding for a single text."""
    print("🔤 Single Text Embedding")
    print("-" * 30)
    
    text = "What are templates in DML programming?"
    print(f"Text: {text}")
    
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=text
    )
    
    embedding = response.data[0]['embedding']
    print(f"✅ Generated {len(embedding)}-dimensional embedding")
    print(f"💰 Used {response.usage.prompt_tokens} tokens")
    print(f"🔢 First 5 values: {embedding[:5]}")
    
    return embedding


async def batch_text_embedding():
    """Generate embeddings for multiple texts."""
    print("\n📦 Batch Text Embedding")
    print("-" * 30)
    
    texts = [
        "DML templates for code reuse",
        "Python functions for modularity", 
        "Reset mechanisms in device modeling",
        "JavaScript modules in web development"
    ]
    
    print(f"Texts ({len(texts)}):")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=texts
    )
    
    embeddings = [item['embedding'] for item in response.data]
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"💰 Used {response.usage.total_tokens} tokens total")
    
    return texts, embeddings


async def semantic_search():
    """Find most similar text to a query."""
    print("\n🔍 Semantic Search")
    print("-" * 30)
    
    query = "How to write reusable code?"
    documents = [
        "DML templates allow code reuse in device modeling",
        "Python classes provide object-oriented programming",
        "Functions help organize code into reusable blocks",
        "Database indexing improves query performance",
        "Template inheritance in DML enables code sharing"
    ]
    
    print(f"Query: {query}")
    print(f"Documents ({len(documents)}):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # Get embeddings for query + documents
    all_texts = [query] + documents
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=all_texts
    )
    
    # Calculate similarities
    query_embedding = response.data[0]['embedding']
    similarities = []
    
    for i, doc_embedding in enumerate(response.data[1:]):
        similarity = cosine_similarity(query_embedding, doc_embedding['embedding'])
        similarities.append((documents[i], similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Most Similar Documents:")
    for i, (doc, sim) in enumerate(similarities[:3], 1):
        print(f"  {i}. [{sim:.3f}] {doc}")


async def custom_parameters():
    """Use custom embedding parameters."""
    print("\n🎛️  Custom Parameters")
    print("-" * 30)
    
    text = "Testing custom embedding parameters"
    
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=text,
        dimensions=512,  # Smaller dimension
        user="example_user",
        encoding_format="float"
    )
    
    embedding = response.data[0]['embedding']
    print(f"✅ Custom embedding generated")
    print(f"📏 Dimensions: {len(embedding)} (requested 512)")
    print(f"💰 Tokens: {response.usage.prompt_tokens}")
    print(f"👤 User: example_user")


def sync_embedding():
    """Synchronous embedding example."""
    print("\n🔄 Synchronous Embedding")
    print("-" * 30)
    
    response = litellm.embedding(
        model="github_copilot/text-embedding-3-small",
        input="Synchronous embedding example"
    )
    
    print(f"✅ Sync embedding: {len(response.data[0]['embedding'])} dimensions")
    print(f"💰 Tokens: {response.usage.prompt_tokens}")


async def main():
    """Run all examples."""
    print("🚀 GitHub Copilot Embedding Examples")
    print("=" * 50)
    
    try:
        # Async examples
        await single_text_embedding()
        await batch_text_embedding()
        await semantic_search()
        await custom_parameters()
        
        # Sync example
        sync_embedding()
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Next steps:")
        print("- Try with your own texts")
        print("- Experiment with different dimensions")
        print("- Build semantic search applications")
        print("- Integrate with your existing workflows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Ensure you have GitHub Copilot access")
        print("- Check your GitHub authentication")
        print("- Verify litellm installation: pip install litellm")


if __name__ == "__main__":
    asyncio.run(main())