# RAG From Scratch

A modular Retrieval-Augmented Generation (RAG) system built without high-level framework
abstractions. Implements parent-child chunking, FAISS vector storage (Flat / IVF / HNSW),
hybrid retrieval (dense + BM25 via Reciprocal Rank Fusion), query expansion, cross-encoder
reranking, and LLM-based answer generation — each component written and controlled manually.

> **Note on "from scratch":** Core retrieval components (chunking, embedding, FAISS indexing,
> sparse retrieval, fusion, reranking, generation) are implemented directly. LangChain's text
> splitter and PDF loader are used as utilities for document I/O, not as a pipeline abstraction.

---

## What This Is

Most RAG tutorials use LangChain or LlamaIndex end-to-end, which hides:

- How chunks are actually stored and retrieved from FAISS
- What parent-child chunking does to retrieval quality
- How dense and lexical (BM25) retrieval are fused without score-scale mismatches
- How cross-encoder reranking differs from dense retrieval, and what it actually buys you
- What "context construction" means before the LLM sees it

This project builds each of those steps explicitly so the internals are visible and modifiable,
and benchmarks them so claims about retrieval quality are backed by numbers instead of names.

---

## Architecture

```
PDF Documents
│
▼
Document Loader              ← PyPDF, handles multi-PDF input
│
▼
Parent-Child Chunker         ← large parent chunks split into smaller child chunks
│
├──────────────────────────────┐
▼                               ▼
Embedding Model                BM25 Index
(dense, instruction-tuned)     (sparse, lexical)
│                               │
▼                               ▼
FAISS Vector Store          Sparse Retriever
(Flat / IVF / HNSW)          (child → parent)
│                               │
▼                               ▼
Dense Retriever  ──────┬──────  │
(child → parent)       ▼        │
                  Hybrid Retriever (RRF fusion)
                        │
                        ▼ (optional) Query Expansion
                        │  generate paraphrases, retrieve each, fuse
                        ▼
                  Cross-Encoder Reranker
                        │
                        ▼
                  Context Builder
                        │
                        ▼
                  LLM Generator
                        │
                        ▼
                      Answer
```

Dense-only vs. hybrid, and query expansion on/off, are config switches — not separate
codepaths — so the same pipeline can be benchmarked across configurations (see Evaluation
below).

---

## Project Structure

```
rag_from_scratch/
│
├── configs/
│   └── config.py              # all configurable parameters (models, chunk sizes, k,
│                               # index type, retrieval mode, RRF k, query expansion)
│
├── data/                      # input PDFs (3 included: Attention, BERT, Multi-Query Attention)
│
├── eval/
│   ├── data_eval.json         # 25 question/expected-(source,page) pairs grounded in
│   │                          # the actual content of the included PDFs
│   ├── evaluator.py           # Hit@K, Recall@K, MRR, NDCG@K
│   └── benchmark.py           # runs the pipeline across multiple configs, reports
│                               # indexing time, retrieval latency, memory footprint
│
├── models/
│   └── model.py                # embedding, reranker, and generation model loading
│
├── rag/
│   ├── loader.py                # PDF ingestion and text extraction
│   ├── chunker.py                # parent-child chunking logic
│   ├── embedder.py               # embedding generation (instruction-prefixed queries)
│   ├── vectorstore.py            # FAISS index construction and persistence
│   ├── retriever.py               # dense top-k retrieval from FAISS
│   ├── sparse_retriever.py        # BM25 lexical retrieval
│   ├── hybrid_retriever.py        # RRF fusion of dense + sparse
│   ├── query_expansion.py         # LLM-based multi-query expansion
│   ├── reranker.py                 # cross-encoder reranking
│   ├── context_builder.py          # assembles retrieved chunks into LLM context
│   ├── generator.py                 # LLM inference and answer generation
│   ├── ingestion.py                  # end-to-end ingestion pipeline
│   └── pipeline.py                    # end-to-end query pipeline, with timing
│
├── main.py                    # entry point (interactive query loop)
├── memory.py                  # memory-footprint tracking utility, used by eval/benchmark.py
├── model_downloading.ipynb    # notebook to pre-download required models from Hugging Face
├── requirements.txt
└── README.md
```

---

## FAISS Index Types

Three index types are supported, selectable via `configs/config.py`:

| Index  | Search Type                 | Speed     | Memory | When to Use                        |
| ------ | ---------------------------- | --------- | ------ | ----------------------------------- |
| `Flat` | Exact (brute force)          | Slow      | High   | Small datasets, need exact results  |
| `IVF`  | Approximate (cluster-based)  | Fast      | Medium | Medium datasets (10k–1M vectors)    |
| `HNSW` | Approximate (graph-based)    | Very fast | High   | Large datasets, low latency needed  |

The included demo corpus (~500 child chunks) is too small for IVF to behave differently from
Flat — IVF clustering needs enough vectors per cluster to matter. The benchmark defaults to
Flat and HNSW; add more PDFs to `data/` if you want a meaningful IVF comparison.

---

## Setup

### Prerequisites

- Python 3.9+
- 8GB RAM minimum (embedding models load into memory)
- GPU optional but recommended for generation

### Install dependencies

```
git clone https://github.com/kapil-chouhan-ai/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```

`requirements.txt` lists unpinned packages so it installs on any OS — after installing, run
`pip freeze > requirements-lock.txt` if you want a reproducible lock for your machine.

### Models used

| Purpose    | Model                                       |
| ---------- | -------------------------------------------- |
| Embeddings | `microsoft/harrier-oss-v1-270m`               |
| Reranking  | `cross-encoder/ms-marco-TinyBERT-L4`          |
| Generation | `dphn/Dolphin3.0-Qwen2.5-0.5B`                 |

`harrier-oss-v1` is instruction-tuned: queries are embedded with a task instruction prepended
(`configs.Config.embed_instruction`), documents are not. This matters — skipping it leaves
retrieval quality below what the model is benchmarked at.

Run `model_downloading.ipynb`, or let `main.py` / `eval/benchmark.py` download them on first run.

---

## Usage

### Run interactively

```
python main.py
```

### Use as a library

```python
from rag.ingestion import IngestionPipeline
from rag.hybrid_retriever import HybridRetriever
from models.model import embed_model, pdfloader

ingestion = IngestionPipeline(loader_model=pdfloader(), embedder_model=embed_model())
artifacts = ingestion.ingest(["data/attention.pdf"])

retriever = HybridRetriever(artifacts["retriever"], artifacts["sparse_retriever"])
results = retriever.retrieve("What is multi-head attention?", k=5)
```

---

## Retrieval Pipeline (Step by Step)

1. Load PDFs → extract raw text per page
2. Split into **parent chunks** (large, ~1000 chars) — preserve context
3. Split parent chunks into **child chunks** (small, ~400 chars) — improve retrieval precision
4. Embed child chunks (instruction-prefixed query side) and index them in FAISS; separately
   index the same child chunks in a BM25 lexical index
5. At query time: retrieve top-k via dense FAISS search **and** BM25, independently
6. **Hybrid fusion**: combine the two ranked lists with Reciprocal Rank Fusion (rank-based,
   so an L2 distance and a BM25 score never need to be put on the same scale)
7. *(optional)* **Query expansion**: generate paraphrases of the query with the LLM, retrieve
   each, fuse the results the same way
8. **Parent expansion**: for each retrieved child, fetch its parent chunk (fuller context)
9. **Cross-encoder reranking**: score each parent chunk against the query, reorder by relevance
10. Build final context string from top reranked chunks
11. Pass context + query to the LLM → generate answer

**Why parent-child?** Child chunks improve retrieval (small, specific), parent chunks improve
generation (large, contextual). You retrieve with small chunks but generate with big ones.

**Why hybrid?** Dense retrieval finds semantically related text even with no shared vocabulary;
BM25 finds exact-term matches dense models sometimes miss (rare terms, acronyms, exact figures).
Fusing both is one of the more reliable ways to lift retrieval quality without changing models.

---

## Evaluation

`eval/data_eval.json` has 25 question/answer-location pairs, each grounded in the actual text
of the three included PDFs (Attention Is All You Need, BERT, and the original Multi-Query
Attention paper) — verified by extracting page text directly, not guessed from memory of these
papers.

```
python -m eval.benchmark
```

Reports, per configuration:

- **Hit@K** — did any relevant page appear in the top K results
- **Recall@K** — fraction of relevant pages found in the top K
- **MRR** — how high the first relevant result ranked, averaged across queries
- **NDCG@K** — rank-aware quality, rewarding relevant results appearing earlier
- **Indexing time** — wall-clock time to chunk, embed, and build the index
- **Retrieval latency** — mean / p95 per-query latency
- **Memory footprint** — process RSS delta (and CUDA allocation, if available) during indexing

Configurations benchmarked by default: dense-only (Flat, no rerank), dense + rerank (Flat),
dense + rerank (HNSW), and hybrid + rerank (Flat). Results are written to
`eval/benchmark_results.json`.

**Caveat:** ground truth is page-level, which is a coarse proxy — a parent chunk doesn't always
align to exactly one PDF page. Treat these numbers as useful for comparing configurations
against each other, not as an absolute accuracy figure.

### Results

<<<<<<< HEAD
| Config | Hit@5 | Recall@5 | MRR | NDCG@5 | Embed (s) | IdxBuild (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| dense_flat_no_rerank | 0.88 | 0.88 | 0.7333 | 0.7702 | 6.812 | 0.005 | 74.61 |
| dense_flat_rerank | 0.88 | 0.88 | 0.78 | 0.8049 | 6.812 | 0.005 | 99.72 |
| dense_hnsw_rerank | 0.88 | 0.88 | 0.78 | 0.8049 | 6.812 | 0.034 | 98.47 |
| hybrid_flat_rerank | 0.92 | 0.92 | 0.7913 | 0.8232 | 6.812 | 0.005 | 106.21 |
=======
Config                Hit@K   Recall@K  MRR     NDCG@K  Embed(s)  IdxBuild(s) Latency(ms)
----------------------------------------------------------------------------------------
dense_flat_no_rerank  0.88    0.88      0.7333  0.7702  6.812     0.005       74.61
dense_flat_rerank     0.88    0.88      0.78    0.8049  6.812     0.005       99.72
dense_hnsw_rerank     0.88    0.88      0.78    0.8049  6.812     0.034       98.47
hybrid_flat_rerank    0.92    0.92      0.7913  0.8232  6.812     0.005       106.21

>>>>>>> 715ca93 (experiments ignored)

---

## Tech Stack

| Component        | Library                       |
| ----------------- | ----------------------------- |
| PDF loading        | PyPDF                         |
| Text splitting      | LangChain Text Splitters      |
| Embeddings           | Sentence Transformers         |
| Sparse retrieval      | rank-bm25                     |
| Vector database        | FAISS                         |
| Reranking                | Hugging Face cross-encoder    |
| Generation                | Hugging Face Transformers     |
| Memory tracking             | psutil                        |

---

## What's Next

- [ ] Larger eval corpus (the included 3 PDFs are enough to validate correctness, not to
      stress-test IVF or get statistically tight latency numbers)
- [ ] Generation-quality evaluation (faithfulness / answer relevance), not just retrieval
- [ ] Agentic RAG: retrieval as a tool inside a ReAct loop

---

## Motivation

Understanding retrieval systems by building them, not by calling framework abstractions — and
backing claims about what improves retrieval (hybrid search, reranking, query expansion) with
benchmark numbers instead of asserting them.
