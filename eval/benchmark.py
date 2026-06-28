import json
import statistics
import time
 
from configs.config import Config
from eval.evaluator import RetrievalEvaluator
from memory import MemoryTracker
from models.model import embed_model, pdfloader, reranker_model
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.hybrid_retriever import HybridRetriever
from rag.loader import DocumentLoader
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.sparse_retriever import BM25Retriever
from rag.vectorstore import VectorStore
 
DOCS = ["data/attention.pdf", "data/bert.pdf", "data/MQA.pdf"]
EVAL_PATH = "eval/data_eval.json"
EVAL_K = 5
 
# Index-type variation, rerank on/off, and dense-vs-hybrid - enough
# spread to substantiate "across multiple configurations" without an
# unreasonable number of full re-ingestions for a 3-document corpus.
# IVF is implemented (configs/config.py, rag/vectorstore.py) but isn't
# benchmarked by default: with ~40 pages of source PDFs there usually
# aren't enough vectors per cluster for IVF to behave differently from
# Flat. Add more PDFs to data/ and an "ivf" entry below if you want that
# comparison to mean anything.
CONFIGS = [
    {"name": "dense_flat_no_rerank", "index_type": "flat", "mode": "dense", "rerank": False},
    {"name": "dense_flat_rerank", "index_type": "flat", "mode": "dense", "rerank": True},
    {"name": "dense_hnsw_rerank", "index_type": "hnsw", "mode": "dense", "rerank": True},
    {"name": "hybrid_flat_rerank", "index_type": "flat", "mode": "hybrid", "rerank": True},
]
 
 
def load_eval_set():
    with open(EVAL_PATH) as f:
        return json.load(f)
 
 
def measure_latency(fn, queries, n_repeats=3):
    samples = []
    for q in queries:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn(q)
            samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "mean_ms": round(statistics.mean(samples), 2),
        "p95_ms": round(samples[max(0, int(0.95 * len(samples)) - 1)], 2),
    }
 
 
def build_shared_artifacts(config, embedder_model_instance):
    """
    Runs document loading, chunking, and embedding exactly once - the
    expensive steps that don't vary across any of the CONFIGS below.
    Builds one FAISS index per distinct index_type actually needed and
    one shared BM25 index (index_type-independent), reusing the same
    embeddings array for every FAISS build.
    """
    loader = DocumentLoader(pdfloader())
    all_docs = []
    for path in DOCS:
        all_docs.extend(loader.load(path))
 
    chunker = Chunker(
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        parent_chunk_size=config.parent_chunk_size,
        parent_chunk_overlap=config.parent_chunk_overlap,
    )
    parents, children, child_to_parent = chunker.split(all_docs)
 
    embedder = Embedder(embedder_model_instance, query_instruction=config.embed_instruction)
 
    tracker = MemoryTracker()
    mem_before_embed = tracker.snapshot()
    t0 = time.time()
    texts = [doc.page_content for doc in children.values()]
    embeddings = embedder.embed_docs(texts)
    embedding_time_s = round(time.time() - t0, 3)
    mem_after_embed = tracker.snapshot()
 
    sparse_retriever = BM25Retriever(children, child_to_parent, parents)
 
    indexes = {}
    for index_type in {cfg["index_type"] for cfg in CONFIGS}:
        mem_before_index = tracker.snapshot()
        t0 = time.time()
 
        vector_store = VectorStore(index_type=index_type)
        vector_store.add(embeddings)
 
        index_build_time_s = round(time.time() - t0, 3)
        mem_after_index = tracker.snapshot()
 
        indexes[index_type] = {
            "retriever": Retriever(embedder, vector_store, child_to_parent, parents),
            "index_build_time_s": index_build_time_s,
            "index_memory_delta_mb": tracker.delta(mem_before_index, mem_after_index),
        }
 
    return {
        "indexes": indexes,
        "sparse_retriever": sparse_retriever,
        "embedding_time_s": embedding_time_s,
        "embedding_memory_delta_mb": tracker.delta(mem_before_embed, mem_after_embed),
    }
 
 
def run_config(cfg, eval_set, shared, reranker_obj, rrf_k):
    index_info = shared["indexes"][cfg["index_type"]]
    dense = index_info["retriever"]
 
    retriever = (
        HybridRetriever(dense, shared["sparse_retriever"], rrf_k=rrf_k)
        if cfg["mode"] == "hybrid"
        else dense
    )
 
    active_reranker = reranker_obj if cfg["rerank"] else None
    evaluator = RetrievalEvaluator(retriever, reranker=active_reranker, rerank_k=EVAL_K)
    metrics = evaluator.evaluate(eval_set, k=EVAL_K)
 
    questions = [s["question"] for s in eval_set]
    latency = measure_latency(lambda q: evaluator._retrieved_keys(q, EVAL_K), questions)
 
    return {
        "config": cfg["name"],
        "embedding_time_s": shared["embedding_time_s"],
        "index_build_time_s": index_info["index_build_time_s"],
        "index_memory_delta_mb": index_info["index_memory_delta_mb"],
        "retrieval_latency_ms": latency,
        **metrics,
    }
 
 
def main():
    config = Config()
    eval_set = load_eval_set()
 
    embedder_model_instance = embed_model()
    reranker_obj = Reranker(reranker_model())
 
    shared = build_shared_artifacts(config, embedder_model_instance)
 
    results = [
        run_config(cfg, eval_set, shared, reranker_obj, config.rrf_k)
        for cfg in CONFIGS
    ]
 
    with open("eval/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
 
    header = (
        f"{'Config':<22}{'Hit@K':<8}{'Recall@K':<10}{'MRR':<8}{'NDCG@K':<8}"
        f"{'Embed(s)':<10}{'IdxBuild(s)':<12}{'Latency(ms)':<12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['config']:<22}{r['Hit@K']:<8}{r['Recall@K']:<10}{r['MRR']:<8}{r['NDCG@K']:<8}"
            f"{r['embedding_time_s']:<10}{r['index_build_time_s']:<12}"
            f"{r['retrieval_latency_ms']['mean_ms']:<12}"
        )
 
    print(f"\nFull results with memory footprint written to eval/benchmark_results.json")
 
 
if __name__ == "__main__":
    main()