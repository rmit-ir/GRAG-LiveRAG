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

## Usage

Run the main script:

```bash
uv run src/main.py
```

Run it with a particular python version

```bash
uv run -p 3.12 src/main.py
```
