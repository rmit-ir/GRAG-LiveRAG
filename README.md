# LiveRAG

A research project for live retrieval-augmented generation.

## Overview

Project management: [rmit-liverag-2025 on Linear](https://linear.app/rmit-liverag-2025/team/RMI/view/kanban-2d49ab9d373f)

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

## Environment Variables

This project uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```bash
# AWS Configuration (required for AWS services and SSM parameter access)
# Note: AWS_REGION must be set to us-east-1 for this project
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key

# Other service configurations can be added as needed
```

A `.env.example` file is provided as a template. Copy it to create your own `.env` file:

```bash
cp .env.example .env
```

And edit the `.env` file with your own values.

## Usage

Run the main script:

```bash
uv run src/main.py
```

Run it with a particular python version

```bash
uv run -p 3.12 src/main.py
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
