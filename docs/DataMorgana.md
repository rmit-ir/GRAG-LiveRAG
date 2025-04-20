# DataMorgana Dataset Generation

This document provides information on how to use the `create_datamorgana_dataset.py` script to generate synthetic question-answer datasets using DataMorgana.

## Overview

DataMorgana is a tool for generating highly customizable and diverse synthetic Q&A benchmarks tailored to RAG applications. It enables detailed configurations of user and question categories and provides control over their distribution within the benchmark.

## Basic Usage

To generate a dataset with default settings (2 questions in TSV format):

```bash
uv run scripts/create_datamorgana_dataset.py

# Print help message to get detailed usage
uv run scripts/create_datamorgana_dataset.py --help
```

## Configuration File

The configuration file is a JSON file that defines question and user categories. The default configuration is located at `scripts/config/datamorgana_config.json`.

You can also use a custom configuration file like this:

```bash
uv run scripts/create_datamorgana_dataset.py --config=path/to/custom_config.json
```

### Structure

```json
{
  "user_categories": [
    {
      "categorization_name": "expertise-categorization",
      "categories": [
        {
          "name": "expert",
          "description": "a specialized user with deep understanding of the corpus",
          "probability": 0.5
        },
        {
          "name": "novice",
          "description": "a regular user with no understanding of specialized terms",
          "probability": 0.5
        }
      ]
    }
  ],
  "question_categories": [
    {
      "categorization_name": "factuality",
      "categories": [
        {
          "name": "factoid",
          "description": "question seeking a specific, concise piece of information",
          "probability": 0.5
        },
        {
          "name": "open-ended",
          "description": "question inviting detailed or exploratory responses",
          "probability": 0.5
        }
      ]
    },
    // Additional categorizations...
  ]
}
```

### Customizing Categories

Each category has the following properties:

- `name`: A short identifier for the category
- `description`: A detailed description of what the category represents
- `probability`: The likelihood of this category being selected (probabilities within a categorization should sum to 1.0)

For multi-document questions, you can add an additional property:

- `is_multi_doc`: Boolean indicating if this category requires multiple documents

## Output Dataset

The generated dataset includes the following columns:

- `qid`: Unique identifier for each question
- `question`: The generated question
- `answer`: The generated answer
- `context`: The context used to generate the question and answer
- `document_ids`: IDs of the documents used
- Question category columns: One column per question categorization (e.g., `question_factuality`)
- User category columns: One column per user categorization (e.g., `user_expertise-categorization`)
- Metadata columns:
  - `generation_timestamp`: When the dataset was generated
  - `question_length`: Length of the question in words
  - `answer_length`: Length of the answer in words
  - `context_length`: Length of the context in words

## Benefits of DataMorgana

- **High Diversity**: Generates questions with diverse lexical, syntactic, and semantic characteristics
- **Customizable**: Easily configure user and question categories to match your specific use case
- **Lightweight**: Simple two-stage process for efficient generation
- **Flexible Output**: Support for multiple output formats (TSV, Excel, Parquet, JSONL)
