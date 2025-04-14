# Active Context: LiveRAG

## Current Technical Focus

The project is currently focused on implementing the core vector search capabilities and integrating with DataMorgana for Q&A generation. The team is working on:

1. Setting up and configuring vector indices with Pinecone and OpenSearch
2. Implementing embedding generation using the E5-base-v2 model
3. Creating a DataMorgana client for generating diverse Q&A pairs
4. Testing the vector search functionality with sample queries

## Recent Technical Changes

- Implemented Pinecone index integration in `src/services/pinecone_index.py`
- Created OpenSearch index integration in `src/services/opensearch_index.py`
- Developed embedding utilities in `src/services/embedding_utils.py`
- Set up AWS utilities for cloud integration in `src/services/aws_utils.py`
- Implemented DataMorgana client in `src/services/ds_data_morgana.py`
- Created path utilities in `src/utils/path_utils.py`
- Set up test notebooks for index evaluation

## Next Technical Steps

1. Implement the complete RAG pipeline by integrating with Falcon3-10B-Instruct LLM
   - Create a new service class `src/services/llm_service.py` for LLM integration
   - Implement prompt template management in `src/services/prompt_utils.py`
   - Develop response generation with 200-token limit enforcement

2. Build evaluation framework for measuring relevance and faithfulness
   - Implement metrics calculation in `src/services/evaluation.py`
   - Create automated testing pipeline in `src/services/test_pipeline.py`

3. Enhance DataMorgana integration for comprehensive testing
   - Generate a dataset of query answer pairs
   - Upload the generated dataset to the SharePoint location: https://rmiteduau-my.sharepoint.com/:x:/r/personal/oleg_zendel_rmit_edu_au/_layouts/15/doc2.aspx?sourcedoc=%7BCAF41455-7A3C-4A27-A0FF-3960E999437D%7D&file=DataMorgana.xlsx&fromShare=true&action=default&mobileredirect=true
   - Extend `src/services/ds_data_morgana.py` with batch processing capabilities
   - Implement result storage and analysis functionality

4. Optimize vector search performance
   - Implement hybrid search combining dense and sparse retrieval
   - Add caching mechanisms for frequently accessed embeddings
   - Create benchmarking tools to compare different retrieval strategies

## Technical Decisions and Considerations

- Evaluating performance tradeoffs between Pinecone and OpenSearch
- Considering optimal embedding strategies for different query types
- Determining the best approach for context selection and prompt construction
- Exploring strategies to optimize for both relevance and faithfulness metrics
- Deciding on the best way to handle the 200-token limit constraint

## Technical Patterns and Preferences

- Modular service-based architecture for flexibility
- Clear separation between index providers through common interfaces
- Jupyter notebooks for interactive testing and demonstration
- Comprehensive testing of vector search capabilities
- Structured error handling and logging

## Technical Learnings and Insights

- Vector search performance varies significantly between providers
- Embedding quality is critical for effective retrieval
- DataMorgana provides valuable diversity in Q&A generation
- The challenge evaluation metrics require careful optimization
