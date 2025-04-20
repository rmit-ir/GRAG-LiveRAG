# LiveRAG

A research project for live retrieval-augmented generation for the SIGIR 2025 LiveRAG Challenge.

## Overview

Project management: [rmit-liverag-2025 on Linear](https://linear.app/rmit-liverag-2025/team/RMI/view/kanban-2d49ab9d373f)

This project is part of the SIGIR 2025 LiveRAG Challenge, which focuses on building effective Retrieval-Augmented Generation systems. The challenge documentation and resources are available in the `hf-space-LiveRAG-challenge` git submodule.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

### Install uv

If you don't have uv installed:

```bash
pip install uv
```

Or using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up the project

1. Clone this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/rmit-ir/LiveRAG
cd LiveRAG
```

If you've already cloned the repository without submodules, you can initialize and update them:

```bash
git submodule init
git submodule update
```

2. Install dependencies

```bash
uv sync
```

3. Setup editable package

So you can import the package from anywhere in the project:

```bash
uv pip install -e .
```

4. Environment variables

A `.env.example` file is provided as a template. Copy it to create your own `.env` file:

```bash
cp .env.example .env
```

And edit the `.env` file with your own values.

Note, the access details can be found at https://linear.app/rmit-liverag-2025/document/access-to-platforms-eb73586791f5

## Usage

Run your scripts:

```bash
uv run scripts/your_script.py
# with a specific python version
uv run -p 3.12 scripts/your_script.py
```

Or notebooks, just open them in VS Code and run them using the python environment from `.venv`.

### Available Scripts and Notebooks

This repository includes several scripts and notebooks for working with the LiveRAG system:

#### Scripts

- [create_datamorgana_dataset.py](scripts/create_datamorgana_dataset.py): Generate synthetic Q&A datasets using DataMorgana

  ```bash
  # Generate 10 questions in TSV format
  uv run scripts/create_datamorgana_dataset.py --n_questions=10
  ```

#### Notebooks

- [test_data_morgana.ipynb](notebooks/test_data_morgana.ipynb): Demonstrates how to use the DataMorgana API for synthetic conversation generation
- [test-indicies.ipynb](notebooks/test-indicies.ipynb): Shows how to use vector index services (Pinecone and OpenSearch) for vector search
- [import_test.ipynb](notebooks/import_test.ipynb): Simple test for package imports

#### Services and Utilities

##### LLM Services

- **BedrockClient**: Amazon Bedrock API client for LLM interactions
  ```python
  from services.llms.bedrock_client import BedrockClient
  
  client = BedrockClient(model_id="anthropic.claude-3-5-haiku-20241022-v1:0")
  response, content = client.query("What is retrieval-augmented generation?")
  ```

- **AI71Client**: AI71 API client for LLM interactions
  ```python
  from services.llms.ai71_client import AI71Client
  
  client = AI71Client(model_id="tiiuae/falcon3-10b-instruct")
  response, content = client.query("What is retrieval-augmented generation?")
  ```

##### Data Generation

- **DataMorgana**: Client for generating synthetic Q&A pairs
  ```python
  from services.ds_data_morgana import DataMorgana
  
  dm = DataMorgana()  # Uses AI71_API_KEY environment variable
  qa_pairs = dm.wait_generation_results(
      dm.generate_qa_pair_bulk(n_questions=10, question_categorizations=[...])["generation_id"]
  )
  ```

##### Vector Search

- **PineconeService**: Client for Pinecone vector database
  ```python
  from services.pinecone_index import PineconeService
  
  service = PineconeService()
  results = service.query_pinecone("What is a second brain?", top_k=3)
  ```

- **OpenSearchService**: Client for OpenSearch vector database
  ```python
  from services.opensearch_index import OpenSearchService
  
  service = OpenSearchService()
  results = service.query_opensearch("What is a second brain?", top_k=3)
  ```

##### Utilities

- **Logging**: Structured logging with context data
  ```python
  from utils.logging_utils import get_logger
  
  logger = get_logger("component_name")
  logger.info("Message with context", key="value", data={"nested": "data"})
  ```

- **Path Utilities**: Helper functions for project paths
  ```python
  from utils.path_utils import get_project_root, get_data_dir
  
  project_root = get_project_root()  # Get absolute path to project root
  data_dir = get_data_dir()  # Get absolute path to data directory
  ```

## Logging

To log messages:

```python
from utils.logging_utils import get_logger

logger = get_logger("component_name")
logger.info("Default info message", context_data={"key": "value"})
logger.debug("Debug message", context_data={"key": "value"})
```

Normally, when running scripts, only info messages will be shown, to see debug messages:

```bash
LOG_LEVEL=DEBUG uv run scripts/your_script.py
```

Or set `LOG_LEVEL=DEBUG` in your `.env` file.

## Dependency Management

Add a dependency

```bash
# uv add <package-name>
uv add pandas
```

## Import System

This project is structured as a Python package installed in editable mode, allowing you to import modules directly.

```bash
# Install the package in editable mode (already done in setup)
uv pip install -e .
```

After installation, you can import services directly:

```python
# Import services in any script or notebook
from services.aws_utils import AWSUtils
from services.pinecone_index import PineconeService
```
