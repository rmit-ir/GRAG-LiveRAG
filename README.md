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

## Environment Variables

This project uses environment variables for configuration. Create a `.env` file in the project root with the following variables:

```
# AWS Configuration (required for AWS services and SSM parameter access)
AWS_REGION=us-east-1 # MUST be this region
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

This project includes a custom import system that allows you to import modules like `from services.aws_utils import AWSUtils` from anywhere in the project.

### How to Import Services in Your Code

#### Option 1: Using the setup_path helper (Recommended)

For scripts at the project root level:

```python
# First import the setup_path module
import setup_path

# Now you can import from services directly
from services.aws_utils import AWSUtils
```

For notebooks in the notebooks directory:

```python
# First import the setup_path module
import setup_path

# Now you can import from services directly
from services.aws_utils import AWSUtils
```

#### Option 2: Direct Import (Only for code inside src directory)

If your code is inside the src directory, you can directly import from services:

```python
from services.aws_utils import AWSUtils
```
