# GitHub Copilot Embedding Examples

This directory contains practical examples demonstrating how to use GitHub Copilot embeddings through the litellm library.

## 📋 Examples Overview

### 1. `github_copilot_embedding_simple.py` - Quick Start
A focused, easy-to-run example covering the most common use cases:
- Single text embedding
- Batch processing
- Semantic search
- Custom parameters
- Synchronous API usage

**Best for**: Getting started quickly and understanding core functionality.

### 2. `github_copilot_embedding_example.py` - Comprehensive Demo
A detailed example showcasing advanced features and use cases:
- Performance comparison with original copilot-api client
- Multiple dimension sizes testing
- Error handling scenarios
- Token usage analysis
- Detailed similarity analysis

**Best for**: Understanding advanced features and integration patterns.

## 🚀 Quick Start

### Prerequisites

1. **GitHub Copilot Subscription**: Individual, Business, or Enterprise
2. **Python Dependencies**:
   ```bash
   pip install litellm numpy
   ```
3. **GitHub Authentication**: The examples will guide you through OAuth if needed

### Run the Simple Example

```bash
python github_copilot_embedding_simple.py
```

Expected output:
```
🚀 GitHub Copilot Embedding Examples
==================================================
🔤 Single Text Embedding
------------------------------
Text: What are templates in DML programming?
✅ Generated 1536-dimensional embedding
💰 Used 7 tokens
🔢 First 5 values: [0.123, -0.456, 0.789, ...]
```

### Run the Comprehensive Example

```bash
python github_copilot_embedding_example.py
```

## 📖 Usage Patterns

### Basic Embedding
```python
import litellm

# Single text
response = await litellm.aembedding(
    model="github_copilot/text-embedding-3-small",
    input="Your text here"
)
embedding = response.data[0]['embedding']

# Multiple texts
response = await litellm.aembedding(
    model="github_copilot/text-embedding-3-small", 
    input=["Text 1", "Text 2", "Text 3"]
)
embeddings = [item['embedding'] for item in response.data]
```

### Custom Parameters
```python
response = await litellm.aembedding(
    model="github_copilot/text-embedding-3-small",
    input="Your text",
    dimensions=512,           # Custom embedding size
    user="your_user_id",      # User tracking
    encoding_format="float"   # Encoding format
)
```

### Synchronous Usage
```python
# For non-async code
response = litellm.embedding(
    model="github_copilot/text-embedding-3-small",
    input="Your text"
)
```

## 🎯 Common Use Cases

### 1. Document Similarity
```python
async def find_similar_documents(query, documents):
    all_texts = [query] + documents
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=all_texts
    )
    
    query_embedding = response.data[0]['embedding']
    similarities = []
    
    for i, doc_data in enumerate(response.data[1:]):
        similarity = cosine_similarity(query_embedding, doc_data['embedding'])
        similarities.append((documents[i], similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)
```

### 2. Batch Processing
```python
async def process_large_dataset(texts, batch_size=100):
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [item['embedding'] for item in response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

### 3. Semantic Search Index
```python
class SemanticIndex:
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    async def add_documents(self, docs):
        response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=docs
        )
        
        self.documents.extend(docs)
        new_embeddings = [item['embedding'] for item in response.data]
        self.embeddings.extend(new_embeddings)
    
    async def search(self, query, top_k=5):
        query_response = await litellm.aembedding(
            model="github_copilot/text-embedding-3-small",
            input=query
        )
        query_embedding = query_response.data[0]['embedding']
        
        similarities = [
            (doc, cosine_similarity(query_embedding, emb))
            for doc, emb in zip(self.documents, self.embeddings)
        ]
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

## 📊 Performance Tips

### 1. Batch Processing
- Process multiple texts together for better efficiency
- Optimal batch size: 50-100 texts per request
- Monitor token usage for cost optimization

### 2. Dimension Selection
- `1536` (default): Full precision, best quality
- `1024`: Good balance of quality and storage
- `512`: Compact, faster similarity calculations

### 3. Caching
```python
import json
from pathlib import Path

def cache_embedding(text, embedding, cache_file="embeddings_cache.json"):
    cache = {}
    if Path(cache_file).exists():
        with open(cache_file) as f:
            cache = json.load(f)
    
    cache[text] = embedding
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

def get_cached_embedding(text, cache_file="embeddings_cache.json"):
    if not Path(cache_file).exists():
        return None
    
    with open(cache_file) as f:
        cache = json.load(f)
    
    return cache.get(text)
```

## 🛠️ Troubleshooting

### Authentication Issues
```
❌ GitHub Copilot authentication failed
```
**Solution**: 
1. Ensure you have an active GitHub Copilot subscription
2. Run the example - it will guide you through OAuth authentication
3. Check that your GitHub account has the necessary permissions

### Model Access Issues
```
❌ Invalid model: github_copilot/text-embedding-3-small
```
**Solution**:
1. Verify your Copilot subscription includes embedding access
2. Try with a different model name if available
3. Check GitHub Copilot service status

### Token Limit Issues
```
❌ prompt token count exceeds the limit
```
**Solution**:
1. Reduce input text length
2. Process texts in smaller batches
3. Split long documents into chunks

### Import Errors
```
❌ No module named 'litellm'
```
**Solution**:
```bash
pip install litellm numpy
```

## 🔗 Integration Examples

### With Vector Databases
```python
# Example with Pinecone
import pinecone

async def index_documents_to_pinecone(documents, index_name):
    # Generate embeddings
    response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=documents
    )
    
    # Prepare vectors for Pinecone
    vectors = []
    for i, item in enumerate(response.data):
        vectors.append({
            "id": f"doc_{i}",
            "values": item['embedding'],
            "metadata": {"text": documents[i]}
        })
    
    # Upsert to Pinecone
    index = pinecone.Index(index_name)
    index.upsert(vectors)
```

### With Existing RAG Systems
```python
# Example RAG integration
async def enhance_rag_with_copilot_embeddings(query, knowledge_base):
    # Generate query embedding
    query_response = await litellm.aembedding(
        model="github_copilot/text-embedding-3-small",
        input=query
    )
    query_embedding = query_response.data[0]['embedding']
    
    # Find relevant documents
    # ... your existing RAG logic here ...
    
    return relevant_documents
```

## 📚 Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [OpenAI Embedding API Reference](https://platform.openai.com/docs/api-reference/embeddings)

## 🤝 Contributing

If you have additional examples or improvements:

1. Fork the repository
2. Add your example with clear documentation
3. Test with real GitHub Copilot credentials
4. Submit a pull request

## 📝 License

These examples are provided under the same license as the litellm project.