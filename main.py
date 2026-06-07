import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import torch
# Fix the missing FP8 attribute bug for older PyTorch versions
if not hasattr(torch, "float8_e8m0fnu"):
    setattr(torch, "float8_e8m0fnu", torch.float32)

# Get currently allocated memory by active tensors (in MB)
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Get total memory being held by the PyTorch cache (in MB)
print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

from models.model import generate_model, embed_model, reranker_model, pdfloader
from rag.ingestion import IngestionPipeline
from rag.pipeline import RAGPipeline
from rag.context_builder import ContextBuilder
from rag.reranker import Reranker
from rag.generator import Generate
from configs.config import Config

config = Config()

ingestion = IngestionPipeline(
    loader_model=pdfloader(),
    embedder_model=embed_model(),
    index_type="flat",
    parent_chunk_size = config.parent_chunk_size,
)

artifacts = ingestion.ingest(
    ["data/attention.pdf", "data/bert.pdf", "data/MQA.pdf"]
)

retriever = artifacts["retriever"]
vector_store = artifacts["vector_store"]

print("---------------ingestion succesful--------------")
print(vector_store.ntotal)

ingestion.save(
    artifacts,
    "store"
)

rag = RAGPipeline(retriever = retriever,
                  reranker = Reranker(reranker_model()),
                  context_builder = ContextBuilder(),
                  generator = Generate(generate_model()))


# Get currently allocated memory by active tensors (in MB)
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Get total memory being held by the PyTorch cache (in MB)
print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


while True:
    query = input("Query: ")
    if query.lower() == "exit":
        break

    result = rag.run(query)
    print(result["answer"])
