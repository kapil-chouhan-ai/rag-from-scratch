# RAG From Scratch

A modular Retrieval-Augmented Generation (RAG) system built without high-level frameworks.
Implements parent-child chunking, FAISS vector storage (Flat / IVF / HNSW), cross-encoder
reranking, and LLM-based answer generation — each component written and controlled manually.

> **Note on "from scratch":** Core retrieval components (chunking, embedding, FAISS indexing,
> reranking, generation) are implemented directly. LangChain's text splitter is used for
> initial document splitting only, as a utility — not as a pipeline abstraction.

---

## What This Is

Most RAG tutorials use LangChain or LlamaIndex end-to-end, which hides:
- How chunks are actually stored and retrieved from FAISS
- What parent-child chunking does to retrieval quality
- How cross-encoder reranking differs from dense retrieval
- What "context construction" actually means before the LLM sees it

This project builds each of those steps explicitly so the internals are visible and modifiable.

---

## Architecture
```
PDF Documents
│
▼
Document Loader          ← PyPDF, handles multi-PDF input
│
▼
Parent-Child Chunker     ← large parent chunks split into smaller child chunks
│
▼
Embedding Model          ← Sentence Transformers (dense embeddings)
│
▼
FAISS Vector Store       ← Flat / IVF / HNSW (configurable)
│
▼
Retriever                ← top-k child chunk retrieval
│
▼
Parent Expansion         ← retrieved child → fetch its parent for fuller context
│
▼
Cross-Encoder Reranker   ← reorders results by relevance to query
│
▼
Context Builder          ← assembles final context string
│
▼
LLM Generator            ← Hugging Face model produces final answer
│
▼
Answer
```
---

## Project Structure
```
rag_from_scratch/
│
├── configs/
│   └── config.py              # all configurable parameters (models, chunk sizes, k, index type)
│
├── data/                      # place input PDF files here
│
├── eval/                      # evaluation scripts for retrieval quality
│
├── experiments/               # exploratory notebooks and ablation tests
│
├── models/
│   └── model.py               # embedding and LLM model loading
│
├── rag/
│   ├── loader.py              # PDF ingestion and text extraction
│   ├── chunker.py             # parent-child chunking logic
│   ├── embedder.py            # embedding generation
│   ├── vectorstore.py         # FAISS index construction and persistence
│   ├── retriever.py           # top-k retrieval from FAISS
│   ├── reranker.py            # cross-encoder reranking
│   ├── context_builder.py     # assembles retrieved chunks into LLM context
│   ├── generator.py           # LLM inference and answer generation
│   ├── ingestion.py           # end-to-end ingestion pipeline
│   └── pipeline.py            # end-to-end query pipeline
│
├── main.py                    # entry point
├── memory.py                  # persistence layer for vector store and document store
├── model_downloading.ipynb    # notebook to download required models from Hugging Face
├── requirements.txt
└── README.md
```
---

## FAISS Index Types

Three index types are supported, selectable via `configs/config.py`:

| Index | Search Type | Speed | Memory | When to Use |
|-------|-------------|-------|--------|-------------|
| `Flat` | Exact (brute force) | Slow | High | Small datasets, need exact results |
| `IVF` | Approximate (cluster-based) | Fast | Medium | Medium datasets (10k–1M vectors) |
| `HNSW` | Approximate (graph-based) | Very fast | High | Large datasets, low latency needed |

For learning purposes, start with `Flat` (exact, no surprises). Switch to `IVF` or `HNSW` when
dataset size makes brute-force retrieval slow.

---

## Setup

### Prerequisites

- Python 3.9+
- 8GB RAM minimum (embedding models load into memory)
- GPU optional but recommended for LLM generation

### Install dependencies

```bash
git clone https://github.com/kapil-18-pythonic/rag_from_scratch.git
cd rag_from_scratch
pip install -r requirements.txt
```

### Download models

Run the notebook to download the required Hugging Face models locally:

```bash
jupyter notebook model_downloading.ipynb
```

Or download manually — the models used are:

| Purpose | Model |
|---------|-------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | `dphn/Dolphin3.0-Qwen2.5-0.5B` |

---

## Usage

### 1. Configure

Edit `configs/config.py` to set:
- PDF paths
- Chunk sizes (parent and child)
- Number of retrieved results (`top_k`)
- FAISS index type (`flat`, `ivf`, or `hnsw`)
- Model names

### 2. Ingest documents

```python
from rag.ingestion import ingest

ingest(pdf_paths=["data/your_document.pdf"])
```

This loads PDFs, chunks them, embeds child chunks, and saves the FAISS index to disk.

### 3. Query

```python
from rag.pipeline import query

answer = query("What is the capital of France?")
print(answer)
```

### 4. Run from terminal

```bash
python main.py
```

---

## Retrieval Pipeline (Step by Step)

1. Load PDFs → extract raw text per page
2. Split into **parent chunks** (large, ~512 tokens) — preserve context
3. Split parent chunks into **child chunks** (small, ~128 tokens) — improve retrieval precision
4. Embed child chunks using Sentence Transformers
5. Store child embeddings in FAISS index
6. At query time: embed query → retrieve top-k child chunks from FAISS
7. **Parent expansion**: for each retrieved child, fetch its parent chunk (larger context)
8. **Cross-encoder reranking**: score each parent chunk against the query, reorder by relevance
9. Build final context string from top reranked chunks
10. Pass context + query to LLM → generate answer

**Why parent-child?** Child chunks improve retrieval (small, specific), parent chunks improve
generation (large, contextual). You retrieve with small chunks but generate with big ones.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| PDF loading | PyPDF |
| Text splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Vector database | FAISS |
| Reranking | Hugging Face cross-encoder |
| Generation | Hugging Face Transformers |
| Persistence | Custom (`memory.py`) |

---

## What's Next

- [ ] Hybrid search: BM25 + dense retrieval with score fusion
- [ ] Query expansion before retrieval
- [ ] Evaluation framework using retrieval metrics (MRR, NDCG, Hit@k)
- [ ] Agentic RAG: retrieval as a tool inside a ReAct loop

---

## Motivation

Understanding retrieval systems by building them, not by calling framework abstractions.
Every component here — chunking, indexing, reranking, context assembly — is explicit and readable.
