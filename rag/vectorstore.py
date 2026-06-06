import faiss
import numpy as np

class VectorStore:

    def __init__(
        self,
        index_type,
        n_list,
        nprobe,
        hnsw_m,
        hnsw_efconstruct,
        hnsw_efsearch
    ):
        self.index_type = index_type
        self.n_list = n_list
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.hnsw_efconstruct = hnsw_efconstruct
        self.hnsw_efsearch = hnsw_efsearch
        self.index = None

    def add(self, embeddings):
        d = embeddings.shape[1]

        if self.index is None:
            if self.index_type == "flat":
                self.index = faiss.IndexFlatL2(d)

            elif self.index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(d, self.hnsw_m)
                self.index.hnsw.efConstruction = self.hnsw_efconstruct
                self.index.hnsw.efSearch = self.hnsw_efsearch

            elif self.index_type == "ivf":
                n_list = min(self.n_list, max(1, int(np.sqrt(len(embeddings)))))
                quantizer = faiss.IndexFlatL2(d)
                self.index = faiss.IndexIVFFlat(quantizer, d, n_list)
                self.index.train(embeddings)
                self.index.nprobe = min(self.nprobe, n_list)

            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

        return self.index.add(embeddings)

    def search(self, query_embeddings, k=5):
        return self.index.search(query_embeddings, k)

    @property
    def ntotal(self):
        return self.index.ntotal