import math


class RetrievalEvaluator:
    """
    Computes Hit@K, Recall@K, MRR, and NDCG@K against a labeled eval set.

    Each eval sample is:
        {"question": str, "expected": [{"source": str, "page": int}, ...]}

    Two real bugs in the original version of this file, both fixed here:
    1. It read doc["metadata"]["page_num"], but PyPDFLoader's metadata
       key is "page" - this raised a KeyError the moment evaluate() was
       actually run against a real retriever.
    2. It compared on page number alone. With a multi-document corpus,
       page 0 of attention.pdf and page 0 of bert.pdf collide - matching
       page alone produces false hits. Ground truth and retrieved docs
       are now compared as (source, page) pairs.

    Pass `reranker` to measure the full retrieve-then-rerank ordering
    instead of raw retrieval - that's what actually substantiates a
    "reranking improves retrieval effectiveness" claim with numbers.

    Caveat: page-level ground truth is a coarse proxy for chunk-level
    retrieval - a parent chunk doesn't always align to exactly one PDF
    page. Treat these metrics as useful for comparing configurations
    against each other, not as an absolute accuracy figure.
    """

    def __init__(self, retriever, reranker=None, rerank_k=5):
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_k = rerank_k

    def _retrieved_keys(self, query, k):
        fetch_k = max(k, self.rerank_k) if self.reranker else k
        docs = self.retriever.retrieve(query, k=fetch_k)

        if self.reranker:
            docs = self.reranker.rerank(query, docs, top_k=k)

        keys = [(doc["metadata"]["source"], doc["metadata"]["page"]) for doc in docs]

        # Dedupe while preserving rank order. Adjacent parent chunks can
        # share the same PDF page (a page's text can get split across
        # two parent chunks), so the same (source, page) key can appear
        # twice in the retrieved list at different ranks. Without this,
        # a single relevant page can be counted as a hit twice in DCG,
        # producing NDCG > 1.0 - caught via integration testing against
        # the real corpus, not visible with k=1-expected-page toy tests.
        seen = set()
        deduped = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)

        return deduped[:k]

    @staticmethod
    def _expected_keys(sample):
        return {(e["source"], e["page"]) for e in sample["expected"]}

    def hit_rate(self, dataset, k=5):
        hits = 0
        for sample in dataset:
            retrieved = self._retrieved_keys(sample["question"], k)
            if self._expected_keys(sample) & set(retrieved):
                hits += 1
        return hits / len(dataset)

    def recall_at_k(self, dataset, k=5):
        recalls = []
        for sample in dataset:
            retrieved = set(self._retrieved_keys(sample["question"], k))
            expected = self._expected_keys(sample)
            recalls.append(len(retrieved & expected) / len(expected))
        return sum(recalls) / len(dataset)

    def mrr(self, dataset, k=10):
        scores = []
        for sample in dataset:
            retrieved = self._retrieved_keys(sample["question"], k)
            expected = self._expected_keys(sample)
            rank = next((i + 1 for i, key in enumerate(retrieved) if key in expected), None)
            scores.append(1.0 / rank if rank else 0.0)
        return sum(scores) / len(dataset)

    def ndcg_at_k(self, dataset, k=5):
        scores = []
        for sample in dataset:
            retrieved = self._retrieved_keys(sample["question"], k)
            expected = self._expected_keys(sample)

            dcg = sum(
                1.0 / math.log2(i + 2) for i, key in enumerate(retrieved) if key in expected
            )
            ideal_hits = min(len(expected), k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
            scores.append(dcg / idcg if idcg > 0 else 0.0)
        return sum(scores) / len(dataset)

    def evaluate(self, dataset, k=5):
        return {
            "Hit@K": round(self.hit_rate(dataset, k), 4),
            "Recall@K": round(self.recall_at_k(dataset, k), 4),
            "MRR": round(self.mrr(dataset, k), 4),
            "NDCG@K": round(self.ndcg_at_k(dataset, k), 4),
        }