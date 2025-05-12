# LiveRAG

A research project for live retrieval-augmented generation for the SIGIR 2025 LiveRAG Challenge.

## Overview

Project management: [rmit-liverag-2025 on Linear](https://linear.app/rmit-liverag-2025/team/RMI/view/kanban-2d49ab9d373f)

This project is part of the SIGIR 2025 LiveRAG Challenge, which focuses on building effective Retrieval-Augmented Generation systems. The challenge documentation and resources are available at <https://huggingface.co/spaces/LiveRAG/Challenge>.

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

1. Clone this repository:

```bash
git clone https://github.com/rmit-ir/LiveRAG
cd LiveRAG
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

Note, the access details can be found at <https://linear.app/rmit-liverag-2025/document/access-to-platforms-eb73586791f5>

## The LiveRAG Workflow

The LiveRAG project follows a structured workflow for developing, running, and evaluating RAG systems:

1. **Create Dataset**: Generate synthetic Q&A pairs using DataMorgana

   ```bash
   uv run scripts/create_datamorgana_dataset.py --n_questions=10
   ```

   The generated dataset will be saved in [data/generated_qa_pairs/](data/generated_qa_pairs/).

   For detailed configuration options, refer to [DataMorgana.md](docs/DataMorgana.md).

   You can also take an existing (recent) dataset from [data/generated_qa_pairs/](data/generated_qa_pairs/) and use it for next step directly.

2. **Run RAG System**: Process questions through a RAG system to generate answers

   ```bash
   # You can use either the full path or just the class name
   uv run scripts/run.py --system BasicRAGSystem \
     --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv \
     --num-threads 5
   ```

   This generates results in [data/rag_results/](data/rag_results/) (git ignored).

   For detailed usage of RAG systems, refer to [systems/README.md](src/systems/).

3. **Evaluate Results**: Compare RAG system outputs against reference answers

   ```bash
   # You can use either the full path or just the class name
   uv run scripts/evaluate.py \
     --evaluator EditDistanceEvaluator \
     --results data/rag_results/dmds_JK09SKjyanxs1_BasicRAGSystem.tsv \
     --reference data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv
   ```

   This generates evaluation results in [data/evaluation_results/](data/evaluation_results/) (git ignored).

   Record your results in [RMIT LiveRAG Systems Test Log.xlsx](https://rmiteduau.sharepoint.com/:x:/r/sites/ComplexQuestionAnswering-LiveRAGSIGIR2025/_layouts/15/Doc2.aspx?action=editNew&sourcedoc=%7B73ac296d-d3cb-461a-be97-469f2252ee1a%7D&wdOrigin=TEAMS-MAGLEV.teamsSdk_ns.rwc&wdExp=TEAMS-TREATMENT&wdhostclicktime=1745570795224&web=1).

   For detailed usage of evaluators, refer to [evaluators/README.md](src/evaluators/README.md).

4. **Analyze & Iterate**: Review evaluation metrics, identify areas for improvement, and refine your RAG system

This workflow enables systematic development and benchmarking of RAG systems, allowing for continuous improvement through iterative refinement.

### Example of "Run System and Evaluate" in one command

With default AI71 LLMs:

```bash
# Run the RAG system over the dataset
uv run scripts/run.py --system BasicRAGSystem --input data/generated_qa_pairs/dmds_fJ20pJnq9zc20.easy.n20.tsv; \
# Evaluate the results
uv run scripts/evaluate.py --evaluator LLMEvaluator --results data/rag_results/dmds_fJ20pJnq9zc20_BasicRAGSystem.tsv --reference data/generated_qa_pairs/dmds_fJ20pJnq9zc20.easy.n20.tsv --num_threads 20
```

With EC2 LLMs (involve starting and stopping EC2 instance), this will take 9min to start the instance:

```bash
# Install AWS CLI plugin, used to ssh into the EC2 instance and set up port forwarding
brew install awscli
brew install --cask session-manager-plugin

# Launch the LLM, it will be available at localhost
# --app-name is required, choose either "vllm" or "mini-tgi"
uv run scripts/aws/deploy_ec2_llm.py --app-name vllm

####################################################
# In a separate terminal, run these as one command
# Wait until the LLM is ready,
uv run scripts/aws/deploy_ec2_llm.py --app-name vllm --wait; \
say 'llm is ready'; \
# Run the RAG system over the dataset
uv run scripts/run.py --system VanillaRAG --input data/generated_qa_pairs/dmds_2_05012333.tsv --num-threads 20 --llm_client ec2_llm; \
# Stop the EC2 LLM instance
uv run scripts/aws/deploy_ec2_llm.py --app-name vllm --stop; \
# Evaluate the results
uv run scripts/evaluate.py --evaluator LLMEvaluator --results data/rag_results/dmds_2_05012333_VanillaRAG.tsv --reference data/generated_qa_pairs/dmds_2_05012333.tsv --num_threads 20; \
say "evaluation finished"
```

## Live Day Procedure

1. Launch EC2 LLM instances

   ```bash
   # terminal 1
   uv run scripts/aws/deploy_ec2_llm.py --app-name vllm
   # or connect to existing instances
   uv run scripts/aws/deploy_ec2_llm.py --app-name vllm --connect i-xxxxxxxxxxxx
   
   # terminal 2
   uv run scripts/aws/deploy_ec2_llm.py --app-name mini-tgi --instance-type g6e.12xlarge
   # or connect to existing instances
   uv run scripts/aws/deploy_ec2_llm.py --app-name mini-tgi --connect i-xxxxxxxxxxxx
   ```

   This will take around 9 minutes to start the instance.
2. Download the jsonl LiveRAG questions, save to [data/live_rag_questions/](data/live_rag_questions/)
3. Run the system over questions, results will be saved to [data/rag_results/](data/rag_results/)

  ```bash
  uv run scripts/run.py --system AnovaRAGLite \
    --live \
    --num-threads 20 \
    --llm_client ec2_llm \
    --query_expansion_mode none \
    --n_queries 8 \
    --query_gen_prompt_level medium \
    --qpp no \
    --initial_retrieval_k_docs 50 \
    --first_step_ranker both_fusion \
    --reranker logits \
    --context_words_limit 15000 \
    --rag_prompt_level naive \
    --input data/live_rag_questions/questions.jsonl
  ```

4. Collect the results under [data/rag_results/](data/rag_results/) and send it to the organizers
5. Optionally, you can evaluate the results (no gold answer, scores are only for reference)

  ```bash
  uv run scripts/evaluate.py --evaluator LLMEvaluator \
    --results data/rag_results/LiveRAG_Dry_Test_Question_file.run05051753.tsv \
    --num_threads 20 \
    --no-use_gold_references \
    --reference data/generated_qa_pairs/fake.tsv
  ```

## Usage

Run your scripts:

```bash
uv run scripts/your_script.py
# with a specific python version
uv run -p 3.12 scripts/your_script.py
```

For notebooks, just open them in VS Code and run them using the python environment from `.venv`.

### Available Scripts and Notebooks

This repository includes several scripts and notebooks for working with the LiveRAG system:

#### Scripts

- [create_datamorgana_dataset.py](scripts/create_datamorgana_dataset.py): Generate synthetic Q&A datasets using DataMorgana
- [run.py](scripts/run.py): Run a specified RAG system on a dataset of questions and save the results
- [evaluate.py](scripts/evaluate.py): Evaluate RAG system results against reference answers (DataMorgana dataset) using specified evaluators
- [deploy_ec2_llm.py](scripts/aws/deploy_ec2_llm.py): Deploy and run LLMs on AWS EC2 instances

#### Notebooks

- [test_data_morgana.ipynb](notebooks/test_data_morgana.ipynb): Demonstrates how to use the DataMorgana API for synthetic conversation generation
- [test-indicies.ipynb](notebooks/test-indicies.ipynb): Shows how to use vector index services (Pinecone and OpenSearch) for vector search
- [import_test.ipynb](notebooks/import_test.ipynb): Simple test for package imports

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

- **EC2 LLMs**: Deploy and use LLMs on AWS EC2 instances

  ```bash
  # The --app-name parameter is required, choose either "vllm" or "mini-tgi"
  # By default, it will deploy tiiuae/falcon3-10b-instruct on a g6e.8xlarge instance
  # More options are available at `--help`
  # Ctrl-C to stop and destroy all resources created
  uv run scripts/aws/deploy_ec2_llm.py --app-name vllm

  # Or, put this at the end of your command or run it separately to destroy all resources created
  uv run scripts/aws/deploy_ec2_llm.py --app-name vllm --stop
  ```

  The script will:
  1. Create a CloudFormation stack with an EC2 instance
  2. Install vLLM on the instance
  3. Download and run the specified model
  4. Set up port forwarding to access the model locally
  5. Provide OpenAI-compatible API endpoints

  ```python
  # Import the client
  from services.llms.general_openai_client import GeneralOpenAIClient
  
  client = GeneralOpenAIClient()
  response, content = client.query("What is retrieval-augmented generation?")
  ```

#### Data Generation

- **DataMorgana**: Client for generating synthetic Q&A pairs

  ```python
  from services.ds_data_morgana import DataMorgana
  
  dm = DataMorgana()  # Uses AI71_API_KEY environment variable
  qa_pairs = dm.wait_generation_results(
      dm.generate_qa_pair_bulk(n_questions=10, question_categorizations=[...])["generation_id"]
  )
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

This project is structured as a Python package installed in editable mode, allowing you to import modules directly.

```bash
# Install the package in editable mode (already done in setup)
uv pip install -e .
```

After installation, you can import services directly:

```python
# Import services in any script or notebook
from services.live_rag_aws_utils import LiveRAGAWSUtils
from services.pinecone_index import PineconeService
```
