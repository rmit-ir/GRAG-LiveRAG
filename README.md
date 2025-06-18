# RMIT-ADM+S at the SIGIR 2025 LiveRAG Challenge

This repository provides the source code for the GRAG approach submitted by the RMIT-ADM+S team at the SIGIR LiveRAG Challenge. You can find the paper describing our approach [here](https://doi.org/10.48550/arXiv.2506.14516). The official challenge documentation and resources are available at <https://liverag.tii.ae>.

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

This command searches for a class named GRAG and runs the input dataset with it, GRAG is located at [src/systems/grag/grag.py](src/systems/grag/grag.py#L21).

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
## Acknowledgements

We thank the SIGIR 2025 LiveRAG Challenge organizers for the opportunity to participate and their support, and the reviewers for their helpful feedback. This research was conducted by the [ARC Centre of Excellence for Automated Decision-Making and Society (ADM+S, CE200100005)](https://www.admscentre.org.au/), and funded fully by the Australian Government through the Australian Research Council and was undertaken with the assistance of computing resources from [RACE (RMIT AWS Cloud Supercomputing)](https://www.rmit.edu.au/partner/hubs/race).
  
  This work was conducted on the unceded lands of the  Woi wurrung and Boon wurrung language groups of the eastern Kulin Nation. We pay our respect to Ancestors and Elders, past and present, and extend that respect to all Aboriginal and Torres Strait Islander peoples today and their connections to land, sea, sky, and community.

## Citation

If you use this resource please cite the following report:

Ran, K., Sun, S., Dinh Anh, K. N., Spina, D., & Zendel, O. RMIT-ADM+S at the SIGIR 2025 LiveRAG Challenge -- GRAG: Generation-Retrieval-Augmented Generation. SIGIR 2025 LiverRAG Challenge. DOI: [10.48550/arXiv.2506.14516](https://doi.org/10.48550/arXiv.2506.14516)

```bibtex
@inproceedings{ran2025grag,
 author = {Ran, Kun and Sun, Shuoqi and Dinh Anh, Khoi Nguyen and Spina, Damiano and Zendel, Oleg},
 title = {RMIT-ADM+S at the SIGIR 2025 LiveRAG Challenge -- GRAG: Generation-Retrieval-Augmented Generation},
 booktitle = {LiveRAG Challenge at SIGIR 2025},
 note = {Chosen as one of the top four finalists.},
 abstract = {This paper presents the RMIT-ADM+S participation in the SIGIR 2025 LiveRAG Challenge. Our Generation-Retrieval-Augmented Generation (GRAG) approach relies on generating a hypothetical answer that is used in the retrieval phase, alongside the original question. GRAG also incorporates a pointwise large language model (LLM)-based re-ranking step prior to final answer generation. We describe the system architecture and the rationale behind our design choices. In particular, a systematic evaluation using the Grid of Points (GoP) framework and N-way ANOVA enabled comparison across multiple configurations, including query variant generation, question decomposition, rank fusion strategies, and prompting techniques for answer generation. Our system achieved a Relevance score of 1.199 and a Faithfulness score of 0.477 on the private leaderboard, placing among the top four finalists in the LiveRAG 2025 Challenge.},
 year = {2025},
 doi = {10.48550/arXiv.2506.14516},
 numpages = {9}
}
```


