class HybridRetriever:
    """
    Fuses a dense retriever and a sparse (BM25) retriever using
    Reciprocal Rank Fusion (RRF).

    Why RRF and not a weighted score sum: dense retrieval scores here are
    L2 distances and BM25 scores are unbounded term-weight sums - the two
    are not on comparable scales, so summing them directly would require
    ad-hoc normalization. RRF sidesteps that by using each document's
    *rank position* in each list instead of its raw score:

        score(d) = sum_over_retrievers( 1 / (rrf_k + rank_of_d) )

    rrf_k=60 is the standard constant from the original RRF paper
    and just dampens the influence of very top
    ranks; results aren't especially sensitive to it.
    """

    def __init__(self, dense_retriever, sparse_retriever, rrf_k=60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query, k=10, candidate_pool=30):
        dense_results = self.dense_retriever.retrieve(query, k=candidate_pool)
        sparse_results = self.sparse_retriever.retrieve(query, k=candidate_pool)

        fused_scores = {}
        doc_lookup = {}

        for rank, doc in enumerate(dense_results):
            pid = doc["parent_id"]
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_lookup[pid] = doc

        for rank, doc in enumerate(sparse_results):
            pid = doc["parent_id"]
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_lookup.setdefault(pid, doc)

        ranked_ids = sorted(fused_scores, 
                            key=lambda pid: fused_scores[pid], 
                            reverse=True)[:k]

        results = []
        for pid in ranked_ids:
            doc = dict(doc_lookup[pid])
            doc["retrieval_score"] = fused_scores[pid]
            results.append(doc)
        return results