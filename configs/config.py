from dataclasses import dataclass

@dataclass
class Config:

    child_chunk_size: int = 400
    child_chunk_overlap: int = 100

    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 200

    index_type: str = "flat"

    n_list: int = 100
    nprobe: int = 10

    hnsw_m: int = 16
    hnsw_efconstruct: int = 40
    hnsw_efsearch: int = 16

    retrieve_k: int = 30
    rerank_k: int = 5