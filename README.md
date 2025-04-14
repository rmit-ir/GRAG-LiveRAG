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

Run the main script:

```bash
uv run src/main.py
```

Run it with a particular python version

```bash
uv run -p 3.12 src/main.py
```

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

## Memory Bank

This project uses a memory bank to maintain comprehensive documentation and project context. The memory bank is located in the `memory-bank` directory and consists of the following core files:

1. `projectbrief.md` - Foundation document defining core requirements and goals
2. `productContext.md` - Why this project exists, problems it solves, and user experience goals
3. `systemPatterns.md` - System architecture, design patterns, and component relationships
4. `techContext.md` - Technologies used, development setup, and technical constraints
5. `activeContext.md` - Current work focus, recent changes, and next steps, ignored by git
6. `progress.md` - What works, what's left to build, and current status, ignored by git

The memory bank serves as the central knowledge repository for the project, allowing for consistent understanding and progress tracking across development sessions. It should be kept up-to-date as the project evolves.
