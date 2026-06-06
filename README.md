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

## Why This Project Matters

Long documents are difficult for standard retrieval systems because relevant information may be buried across many sections. This project demonstrates a practical retrieval pipeline that improves long-document search through chunking, vector search, and fine-grained re-ranking.

It reflects my interest in building AI systems for retrieval, ranking, long-document understanding, and practical LLM-powered applications.

## Note

This repository contains a cleaned project write-up and selected implementation components. Course-specific materials, assignment instructions, and private datasets are not included.
