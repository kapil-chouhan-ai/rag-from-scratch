# RAG From Scratch

A modular Retrieval-Augmented Generation (RAG) system built from scratch using FAISS, Sentence Transformers, and Hugging Face models.

## Features

* Parent-Child Chunking
* Multi-PDF Ingestion
* FAISS Vector Store

  * Flat Index
  * IVF Index
  * HNSW Index
* Dense Retrieval
* Cross-Encoder Reranking
* Context Builder
* LLM-based Answer Generation
* Persistence Support
* Modular Architecture

## Architecture

PDF Documents
→ Parent-Child Chunking
→ Embeddings
→ FAISS Index
→ Retriever
→ Reranker
→ Context Builder
→ Generator (LLM)

## Project Structure

```text
rag_from_scratch/
├── configs/
├── experiments/
├── models/
├── rag/
├── main.py
├── memory.py
├── requirements.txt
└── README.md
```

## Tech Stack

* Python
* FAISS
* Sentence Transformers
* Hugging Face Transformers
* LangChain Text Splitters

## Future Improvements

* Hybrid Search (BM25 + Dense Retrieval)
* Query Expansion
* Retrieval Evaluation Framework
* Distillation Experiments
* Advanced RAG Techniques

## Usage

```bash
pip install -r requirements.txt
python main.py
```
