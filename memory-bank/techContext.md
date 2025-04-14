# Technical Context: LiveRAG

## Technologies Used

### Core Technologies

- **Python**: Primary programming language (requires Python 3.11+)
- **Vector Databases**:
  - **Pinecone**: Cloud-based vector database service
  - **OpenSearch**: Search and analytics engine with vector search capabilities
- **Embedding Models**: 
  - **E5-base-v2**: Default embedding model for text-to-vector conversion
- **Large Language Models**:
  - **Falcon3-10B-Instruct**: LLM for generating responses
- **AWS**: Cloud infrastructure for deployment and scaling
- **DataMorgana**: Tool for generating diverse Q&A benchmarks, Markdown version of the original paper is under docs/data-morgana.md

### Libraries and Frameworks

- **Jupyter Notebooks**: For interactive development and testing
- **PyTorch**: For machine learning operations and model inference
- **Transformers**: For loading and using embedding and language models
- **Boto3**: For AWS service integration
- **Python-dotenv**: For environment variable management
- **Pandas**: For data manipulation and analysis

## Development Setup

- Project uses Python with a structured package layout
- Dependencies managed via `pyproject.toml` and `uv.lock` (see `pyproject.toml` for specific dependencies)
- Environment variables configured through `.env` files (see `.env.example` for required variables)
- Python version specified in `.python-version` file
- Package installed in editable mode for easy imports

## Technical Constraints

- Vector database performance limitations
- Embedding model size and computational requirements
- API rate limits for external services
- Cost considerations for cloud-based vector databases
- 200-token limit for RAG responses in the challenge
- Evaluation metrics focused on relevance and faithfulness

## Import System

The project is structured as a Python package that can be installed in editable mode:

```bash
uv pip install -e .
```

This allows importing project modules from anywhere:

```python
from services.aws_utils import AWSUtils
from services.pinecone_index import PineconeService
```

The package structure is defined in `pyproject.toml` with this configuration:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

This tells setuptools to look for packages in the `src` directory, making the packages inside `src/` directly importable.

## Project Requirements

1. Implement efficient vector search capabilities using multiple index providers (Pinecone, OpenSearch)
2. Create embedding utilities for converting text to vector representations
3. Build a system that can generate and evaluate question-answer pairs using DataMorgana
4. Integrate with Falcon3-10B-Instruct LLM for answer generation
5. Optimize for both relevance and faithfulness in RAG responses
6. Provide comprehensive evaluation tools for RAG performance

## Reference Notebooks

The following notebooks provide examples and documentation for using key services:

### HF Space LiveRAG Challenge Reference Notebooks

- **`hf-space-LiveRAG-challenge/Operational_Instructions/DM_API_usage_example.ipynb`**:  
  Demonstrates DataMorgana API usage including:
  - Checking API budget
  - Generating single and bulk Q&A pairs
  - Creating custom question and user categorizations
  - Working with multi-document questions
  - Retrieving and processing generation results

- **`hf-space-LiveRAG-challenge/Operational_Instructions/Indices_Usage_Examples_for_LiveRAG.ipynb`**:  
  Shows how to use pre-built vector indices including:
  - AWS setup and credential configuration
  - Pinecone vector search implementation
  - OpenSearch vector search implementation
  - Batch query processing for both services
  - Text embedding generation with E5-base-v2

### Project Implementation Notebooks

- **`notebooks/test_data_morgana.ipynb`**:  
  Tests the project's DataMorgana service implementation:
  - Synchronous Q&A pair generation
  - Bulk generation with multiple categorizations
  - Working with default and custom categories
  - Parsing results from JSONL files
  - Converting results to DataFrame format

- **`notebooks/test-indicies.ipynb`**:  
  Tests the project's vector index services:
  - AWS utilities for parameter and secret retrieval
  - Embedding generation with hardware acceleration
  - Pinecone service for vector search
  - OpenSearch service for vector search
  - Batch query processing
