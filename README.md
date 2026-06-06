# RAG Retrieval and ColBERT-style Re-ranking System

A retrieval-focused NLP project for searching long meeting transcripts using dense vector retrieval and ColBERT-style token-level re-ranking.

## Overview

This project builds a retrieval pipeline for long-form meeting transcripts. The system splits transcripts into overlapping chunks, embeds them with transformer-based models, indexes them using FAISS, retrieves candidate passages with semantic search, and improves ranking quality with a ColBERT-style re-ranking stage.

The goal is to explore how modern RAG systems combine chunking, embeddings, vector search, re-ranking, and evaluation to surface relevant information from long documents.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- LangChain
- BERT

## What I Built

- Processed long meeting transcripts into overlapping text chunks to improve retrieval granularity.
- Built a dense vector retrieval pipeline using transformer-based embeddings and FAISS.
- Implemented top-k semantic search over transcript chunks.
- Added a ColBERT-style re-ranking stage using BERT token embeddings and token-level max-sim interactions.
- Compared retrieval behavior before and after re-ranking to evaluate ranking improvements.
- Created a small runnable example demonstrating dense retrieval followed by token-level re-ranking.

## Architecture

```text
Long transcripts
      |
      v
Text chunking
      |
      v
Embedding generation
      |
      v
FAISS vector index
      |
      v
Top-k dense retrieval
      |
      v
ColBERT-style token-level re-ranking
      |
      v
Ranked relevant passages
```

## Repository Structure

```text
rag-retrieval-reranking/
├── README.md
├── requirements.txt
└── src/
    ├── chunking.py
    ├── retrieval.py
    ├── reranking.py
    └── example.py
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/rilon75/rag-retrieval-reranking.git
cd rag-retrieval-reranking
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the example:

```bash
python src/example.py
```

The example builds a small retrieval index, retrieves candidate passages with dense vector search, and then re-ranks the results using a ColBERT-style token-level interaction score.

## Implementation Details

### Chunking

Long transcripts are split into overlapping chunks so that retrieval can operate on smaller, more focused passages rather than entire documents. This improves retrieval granularity and helps reduce context-length issues.

### Dense Retrieval

The retrieval stage uses transformer-based sentence embeddings and FAISS vector search. Document chunks are embedded, normalized, and indexed. User queries are embedded in the same vector space and matched against the index using similarity search.

### ColBERT-style Re-ranking

The re-ranking stage uses BERT token embeddings to compare the query and each candidate passage at the token level. For each query token, the system finds the most similar passage token and sums these maximum similarities to produce a late-interaction relevance score.

This allows the system to capture more fine-grained token-level matches than standard dense retrieval alone.

## Why This Project Matters

Long documents are difficult for standard retrieval systems because relevant information may be buried across many sections. This project demonstrates a practical retrieval pipeline that improves long-document search through chunking, vector search, and fine-grained re-ranking.

It reflects my interest in building AI systems for retrieval, ranking, long-document understanding, and practical LLM-powered applications.

## What I Learned

Through this project, I gained hands-on experience with:

- Building retrieval pipelines for long-form text.
- Working with embeddings and vector search.
- Using FAISS for efficient semantic retrieval.
- Applying BERT embeddings for token-level re-ranking.
- Understanding trade-offs between retrieval speed, ranking quality, and model complexity.
- Designing components that resemble the retrieval layer of modern RAG systems.

## Note

This repository contains a cleaned project write-up and selected implementation components. Course-specific materials, assignment instructions, and private datasets are not included.
