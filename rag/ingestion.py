import os
import pickle
import faiss
import time
from rag.loader import DocumentLoader
from rag.chunker import Chunker
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever


class IngestionPipeline:

    def __init__(
        self,
        loader_model,
        embedder_model,
        child_chunk_size=400,
        child_chunk_overlap=100,
        parent_chunk_size=1000,
        parent_chunk_overlap=200,
        index_type="flat",
        n_list=100,
        nprobe=10,
        hnsw_m=16,
        hnsw_efconstruct=40,
        hnsw_efsearch=16
    ):
        self.loader = DocumentLoader(loader_model)
        self.chunker = Chunker(
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap
        )
        self.embedder = Embedder(embedder_model)
        self.vector_store = VectorStore(
            index_type=index_type,
            n_list=n_list,
            nprobe=nprobe,
            hnsw_m=hnsw_m,
            hnsw_efconstruct=hnsw_efconstruct,
            hnsw_efsearch=hnsw_efsearch
        )

    def ingest(self, paths):
        start = time.time()
        print("doc loading starts")
        all_docs = []
        for path in paths:
            all_docs.extend(self.loader.load(path))
        documents = all_docs
        parents, children, child_to_parent = self.chunker.split(documents)

        texts = [doc.page_content for doc in children.values()]
        print(f"--------------chunking succesful--------------------time = {time.time() - start}")
        embeddings = self.embedder.embed_docs(texts)
        self.vector_store.add(embeddings)

        retriever = Retriever(
            self.embedder,
            self.vector_store,
            child_to_parent,
            parents
        )

        return {
            "retriever": retriever,
            "vector_store": self.vector_store,
            "parent_chunks": parents,
            "child_to_parent": child_to_parent
        }


    def save(self, artifacts, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(
            artifacts["vector_store"].index,
            os.path.join(save_dir, "index.faiss")
        )

        with open(os.path.join(save_dir, "parents.pkl"),"wb") as f:
            pickle.dump(artifacts["parent_chunks"], f)

        with open(os.path.join(save_dir, "child_to_parent.pkl"),"wb") as f:
            pickle.dump(artifacts["child_to_parent"], f)

        print(f"Artifacts saved to {save_dir}")

    def load(self, save_dir):
        index = faiss.read_index(os.path.join(save_dir, "index.faiss"))

        with open(os.path.join(save_dir, "parents.pkl"),"rb") as f:
            parents = pickle.load(f)

        with open(os.path.join(save_dir, "child_to_parent.pkl"),"rb") as f:
            child_to_parent = pickle.load(f)

        vector_store = VectorStore(index_type=self.vector_store.index_type)
        vector_store.index = index

        retriever = Retriever(
            self.embedder,
            vector_store,
            child_to_parent,
            parents
        )

        return {
            "retriever": retriever,
            "vector_store": vector_store,
            "parent_chunks": parents,
            "child_to_parent": child_to_parent
        }