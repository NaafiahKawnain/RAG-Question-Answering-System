# RAG-Question-Answering-System

A Retrieval-Augmented Generation (RAG) pipeline that answers questions using retrieved context from Wikipedia passages, rather than relying on an LLM's general knowledge alone.

## Overview
This project implements the full RAG pattern: documents are chunked, embedded, and stored in a vector index. At query time, the most relevant chunks are retrieved and passed to an LLM, which generates an answer grounded in that retrieved context — and explicitly says so when the answer isn't found in the available context, rather than hallucinating.

## Tech Stack
- **Dataset**: `rag-datasets/rag-mini-wikipedia` (Hugging Face)
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector Store**: FAISS
- **LLM**: Groq API (Llama/GPT-OSS models)
- **Language**: Python (Jupyter Notebook)

## Pipeline
1. **Load** — Load passages from the `rag-mini-wikipedia` dataset
2. **Chunk** — Split/prepare passages for embedding
3. **Embed** — Convert chunks into vector embeddings using `all-MiniLM-L6-v2`
4. **Store** — Index embeddings in FAISS for similarity search
5. **Retrieve** — Given a question, embed it and retrieve top-k similar chunks
6. **Generate** — Pass retrieved context + question to Groq LLM to generate a grounded answer

## Example

**Question:** What is the anatomy of a beetle?

**Answer:** *(generated using retrieved context — see notebook for full output)*

The system correctly retrieves relevant passages when available, and responds that the answer isn't in the context when retrieval similarity is low — avoiding hallucination.

## Key Design Choices
- **CPU-only embeddings** — no GPU dependency, runs anywhere
- **Local vector store (FAISS)** — no external DB/service needed
- **Explicit "not found" handling** — the prompt instructs the LLM to only answer from context, and to say so when it can't

## Limitations
- Small, fixed document set — no dynamic document ingestion
- No re-ranking of retrieved chunks beyond similarity search
- No handling of multi-hop questions requiring reasoning across multiple retrieved chunks

## How to Run
1. Install dependencies: `sentence-transformers`, `faiss-cpu`, `groq`, `datasets`
2. Add your Groq API key in the notebook (`groq_client = Groq(api_key="YOUR_API_KEY")`)
3. Run all cells in `RAG.ipynb` sequentially