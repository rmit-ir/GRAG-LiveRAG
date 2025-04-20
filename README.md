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

This project is structured as a Python package that you can install in editable mode, allowing you to import project modules from anywhere.

### Install the Package in Editable Mode

Before using any scripts or notebooks, install the package in editable mode:

```bash
# From the project root directory
uv pip install -e .
```

### How the Import System Works

The package structure is defined in `pyproject.toml` with this configuration:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

This tells setuptools to look for packages in the `src` directory. When the package is installed with `uv pip install -e .`:

1. Python adds the project to your Python path
2. The packages inside `src/` become directly importable
3. `src/services/` becomes available as simply `services` in your imports

### How to Import Services in Your Code

Once the package is installed in editable mode, you can directly import services from anywhere in the project:

```python
# In any script or notebook
from services.aws_utils import AWSUtils
from services.pinecone_index import PineconeService
```

No path manipulation or special import helpers are needed. This approach works consistently across all project files, including notebooks.
