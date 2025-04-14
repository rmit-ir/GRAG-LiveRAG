# System Patterns: LiveRAG

## Architecture Overview

LiveRAG follows a modular architecture with the following key components:

```plain
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Embedding  │────▶│   Vector    │
│ Processing  │     │   Service   │     │   Search    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Output    │◀────│    LLM      │◀────│  Context    │
│ Generation  │     │ Integration │     │ Processing  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Design Patterns

1. **Service-Oriented Architecture**: Each component (embedding, vector search, etc.) is implemented as a separate service
2. **Strategy Pattern**: Multiple interchangeable vector database implementations (Pinecone, OpenSearch)
3. **Factory Pattern**: Creation of appropriate index and embedding services based on configuration
4. **Repository Pattern**: Abstraction layer for data access operations
5. **Dependency Injection**: Services receive their dependencies rather than creating them

## Component Relationships

- **Services Module**: Contains core functionality implementations
  - `pinecone_index.py`: Pinecone vector database integration
  - `opensearch_index.py`: OpenSearch vector database integration
  - `embedding_utils.py`: Text-to-vector embedding utilities
  - `aws_utils.py`: AWS integration services
  - `indicies.py`: Common interface for vector indices
  - `ds_data_morgana.py`: DataMorgana integration for Q&A generation

- **Utils Module**: Contains helper utilities
  - `path_utils.py`: File and directory path management

## Critical Implementation Paths

1. **Vector Search Path**:
   - Query processing → Text embedding → Vector search → Result retrieval
   - Implemented in `pinecone_index.py` and `opensearch_index.py`
   - Uses `embedding_utils.py` for text-to-vector conversion

2. **Q&A Generation Path**:
   - Document selection → Category configuration → DataMorgana API call → Q&A pair generation
   - Implemented in `ds_data_morgana.py`
   - Supports both synchronous and asynchronous generation

3. **RAG Response Path** (to be implemented):
   - Query embedding → Vector search → Context retrieval → LLM prompt construction → Response generation
   - Will integrate Falcon3-10B-Instruct LLM for response generation
   - Will optimize for both relevance and faithfulness metrics

## Data Flow

1. **Input Processing**:
   - User query is received
   - Query is embedded using the embedding model (e.g., E5-base-v2)
   - Embedded query is used to search vector databases

2. **Vector Search**:
   - Vector databases (Pinecone, OpenSearch) are queried with the embedded query
   - Top-k most relevant documents/chunks are retrieved
   - Results include both vector similarity scores and metadata

3. **Context Processing**:
   - Retrieved documents are processed and formatted
   - Most relevant context is selected based on similarity scores
   - Context is prepared for inclusion in the LLM prompt

4. **Response Generation**:
   - Context is combined with the original query in a prompt
   - Prompt is sent to the LLM (Falcon3-10B-Instruct)
   - LLM generates a response based on the provided context
   - Response is formatted to meet the 200-token limit requirement

## Evaluation Flow

1. **Q&A Generation**:
   - DataMorgana is used to generate diverse Q&A pairs
   - Q&A pairs are categorized by question type and user category
   - Generated pairs are stored for evaluation

2. **RAG Evaluation**:
   - RAG system processes the generated questions
   - Responses are evaluated for relevance and faithfulness
   - Metrics are calculated and analyzed
