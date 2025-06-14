# LiveRAG Public Release

A research project for the SIGIR 2025 LiveRAG Challenge.

## Overview

This project focuses on building effective Retrieval-Augmented Generation systems. The official challenge documentation and resources are available at <https://liverag.tii.ae>.

## Reproducing Results

Install uv (alternative installation methods available at [uv docs](https://docs.astral.sh/uv/getting-started/installation/) if you find `curl | sh` unsafe):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Launch the logits server(preferably a GPU instance for faster speed)

```bash
uv run scripts/aws/apps/mini_tgi/llm_server.py --port 8977
```

Configure AI71 credentials

```bash
cp .env.example .env
# edit AI71_API_KEY=
```

Run the final selected config:

```bash
uv run scripts/run.py --system GRAG \
  --live \
  --num-threads 20 \
  --query_expansion_mode none \
  --n_queries 8 \
  --query_gen_prompt_level medium \
  --enable_hyde \
  --qpp no \
  --initial_retrieval_k_docs 50 \
  --first_step_ranker both_fusion \
  --reranker logits \
  --context_words_limit 10000 \
  --rag_prompt_level naive \
  --input data/live_rag_questions/LiveRAG_LCD_Session1_Question_file.jsonl
```

After it finishes, you can find the results in `data/rag_results/` folder.

Note

1. If you hit AI71 rate limits, you can reduce `--num-threads`.
2. By default run.py will connect to logits server at <http://localhost:8977>, if you launch it elsewhere, you need to port forward it to localhost.
3. Error: "An error occurred (UnrecognizedClientException) when calling the GetParameter operation: The security token included in the request is invalid." means you didn't configure the AWS_LIVE_RAG access keys properly.

## General Usage

Run your scripts:

```bash
uv run scripts/your_script.py
# with a specific python version
uv run -p 3.12 scripts/your_script.py
```

For notebooks, just open them in VS Code and run them using the python environment from `.venv`.

### Available Scripts and Utilities

This repository includes several scripts for working with the LiveRAG system:

#### Scripts

- [run.py](scripts/run.py): Run a specified RAG system on a dataset of questions and save the results
- [evaluate.py](scripts/evaluate.py): Evaluate RAG system results against reference answers using various evaluators

### Services and Utilities

#### LLM Services

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

#### Vector Search

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

#### Utilities

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

## Debugging

If you want to debug your code, you can use the VS Code debugger. Refer to [.vscode/launch.json](.vscode/launch.json) for an example configuration.

Then you can just add your breakpoints in the code and run the debugger in VS Code.

## Dependency Management

Add a dependency

```bash
# uv add <package-name>
uv add pandas
```

## Import System

This project is structured as a Python package installed in editable mode, allowing you to import modules directly:

```python
# Import services in any script or notebook
from services.live_rag_aws_utils import LiveRAGAWSUtils
from services.pinecone_index import PineconeService
```
