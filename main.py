# NOTE: the previous version of this file permanently monkey-patched
# torch.float8_e8m0fnu as a workaround for a local torch/transformers
# version mismatch. That's an environment fix, not something that
# belongs in a portfolio project's entry point - pin compatible
# torch/transformers versions in requirements.txt instead (see README).

from models.model import generate_model, embed_model, reranker_model, pdfloader
from rag.ingestion import IngestionPipeline
from rag.pipeline import RAGPipeline
from rag.context_builder import ContextBuilder
from rag.reranker import Reranker
from rag.generator import Generate
from rag.hybrid_retriever import HybridRetriever
from rag.query_expansion import QueryExpander
from configs.config import Config

config = Config()

ingestion = IngestionPipeline(
    loader_model=pdfloader(),
    embedder_model=embed_model(),
    child_chunk_size=config.child_chunk_size,
    child_chunk_overlap=config.child_chunk_overlap,
    parent_chunk_size=config.parent_chunk_size,
    parent_chunk_overlap=config.parent_chunk_overlap,
    index_type=config.index_type,
    n_list=config.n_list,
    nprobe=config.nprobe,
    hnsw_m=config.hnsw_m,
    hnsw_efconstruct=config.hnsw_efconstruct,
    hnsw_efsearch=config.hnsw_efsearch,
    query_instruction=config.embed_instruction,
)

artifacts = ingestion.ingest(["data/attention.pdf", "data/bert.pdf", "data/MQA.pdf"])

ingestion.save(artifacts, "store")

dense_retriever = artifacts["retriever"]
if config.retrieval_mode == "hybrid":
    retriever = HybridRetriever(dense_retriever, artifacts["sparse_retriever"], rrf_k=config.rrf_k)
else:
    retriever = dense_retriever

llm = generate_model()

query_expander = None
if config.use_query_expansion:
    query_expander = QueryExpander(llm, num_variants=config.num_query_variants)

rag = RAGPipeline(
    retriever=retriever,
    reranker=Reranker(reranker_model()),
    context_builder=ContextBuilder(),
    generator=Generate(llm),
    query_expander=query_expander,
)

while True:
    query = input("Query: ")
    if query.lower() == "exit":
        break

    result = rag.run(query, retrieve_k=config.retrieve_k, rerank_k=config.rerank_k, return_timing=True)
    print(result["answer"])
    print(f"  ({result['timing_ms']['total']} ms total)")