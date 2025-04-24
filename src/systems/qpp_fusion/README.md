# QPP Fusion RAG System

An advanced RAG implementation that uses Query Performance Prediction (QPP) to select the most effective queries from multiple generated queries and applies fusion search. The system generates both sparse queries (optimized for keyword search) and dense queries (optimized for semantic search), evaluates their effectiveness using QPP metrics, and selects only the most promising queries for the final retrieval.

## How to Run

```bash
uv run scripts/run.py --system systems.qpp_fusion.qpp_fusion_rag.QPPFusionSystem --input data/generated_qa_pairs/dmds_JK09SKjyanxs1.n5.tsv
```

For help with command-line options:

```bash
uv run scripts/run.py --system systems.qpp_fusion.qpp_fusion_rag.QPPFusionSystem --help
```

## Architecture

```mermaid
graph TD
    A[User Question] --> B[Generate Multiple Queries]
    B --> |Sparse Queries| C[Query Evaluation]
    B --> |Dense Queries| C
    B --> |Original Query| C
    C --> D[Select Most Effective Queries]
    D --> E1[Keyword Search]
    D --> E2[Embedding Search]
    E1 --> F[Fusion of Search Results]
    E2 --> F
    F --> G[Extract Top Documents]
    G --> H[Create Context]
    H --> I[Generate Answer with LLM]
    A --> I
    I --> J[Final Answer]

    style A fill:#f9d5e5
    style J fill:#d5f9e6
    style F fill:#d5e5f9
    style D fill:#f9e6d5
```
