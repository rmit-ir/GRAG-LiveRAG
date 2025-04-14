# Technical Project Brief: LiveRAG

## Overview

LiveRAG is a technical implementation of a Retrieval-Augmented Generation (RAG) system for the SIGIR 2025 LiveRAG Challenge. The system integrates vector databases (Pinecone, OpenSearch) with embedding models to provide accurate and up-to-date information retrieval capabilities.

## Technical Requirements

1. Implement efficient vector search capabilities using multiple index providers (Pinecone, OpenSearch)
2. Create embedding utilities for converting text to vector representations
3. Build a system that can generate and evaluate question-answer pairs using DataMorgana
4. Integrate with Falcon3-10B-Instruct LLM for answer generation
5. Optimize for both relevance and faithfulness in RAG responses
6. Provide comprehensive evaluation tools for RAG performance

## Technical Goals

1. Create a competitive RAG system for the SIGIR 2025 LiveRAG Challenge
2. Support multiple vector database backends (Pinecone, OpenSearch)
3. Optimize for both accuracy and performance in information retrieval
4. Enable real-time updates to the knowledge base
5. Generate diverse and high-quality question-answer pairs using DataMorgana
6. Achieve high scores on the evaluation metrics (relevance and faithfulness)

## Technical Resources

1. Pre-built indices for the FineWeb Sample-10BT corpus (Pinecone and OpenSearch)
2. Access to DataMorgana for generating diverse Q&A benchmarks
3. Access to Falcon3-10B-Instruct LLM for answer generation

## Technical Constraints

1. Must exclusively use Falcon3-10B-Instruct for answer generation
2. Response length limited to 200 tokens
3. Evaluation based on relevance and faithfulness metrics
