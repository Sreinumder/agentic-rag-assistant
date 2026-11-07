# RAG Architecture Guide

## What is RAG?
Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.
It grounds LLM responses in factual, retrieved documents.

## RAG Pipeline Components

### 1. Document Ingestion
- Load documents from various sources (PDF, TXT, MD, etc.)
- Extract text content
- Clean and normalize text

### 2. Chunking
- Split documents into smaller, overlapping chunks
- Typical chunk size: 256-1024 tokens
- Overlap: 10-20% of chunk size to maintain context

### 3. Embedding & Vectorization
- Convert text chunks to dense vector representations
- Use embedding models (e.g., text-embedding-004, Ada-002)
- Store vectors in a vector database

### 4. Retrieval
- Convert user query to embedding
- Find top-K similar chunks using cosine similarity
- Return relevant context

### 5. Generation
- Combine retrieved context with user query
- Send to LLM for response generation
- LLM grounds its answer in the retrieved documents

## Vector Databases
- **ChromaDB**: Lightweight, embedded, good for prototyping
- **Pinecone**: Managed, scalable, cloud-native
- **Weaviate**: Open-source, feature-rich
- **Qdrant**: High-performance, Rust-based
- **FAISS**: Meta's library, optimized for similarity search

## Best Practices
1. Use meaningful chunk sizes for your use case
2. Include metadata (source, date, section) with chunks
3. Implement hybrid search (semantic + keyword) when possible
4. Re-rank retrieved results for better relevance
5. Monitor retrieval quality with evaluation metrics
