import re
from rank_bm25 import BM25Okapi


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """
    Sparse lexical retriever over the same child chunks used for dense
    retrieval. Built once at ingestion time from the `children` dict
    produced by Chunker.split().

    Aggregates child-level BM25 scores up to the parent level the same
    way the dense Retriever does (best child score per parent), so both
    retrievers return comparable parent-ranked lists that HybridRetriever
    can fuse.
    """

    def __init__(self, children, child_to_parent, parents):
        self.child_ids = list(children.keys())
        self.child_to_parent = child_to_parent
        self.parents = parents

        corpus = [_tokenize(children[cid].page_content) for cid in self.child_ids]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query, k=10):
        scores = self.bm25.get_scores(_tokenize(query))

        parent_best = {}
        for child_idx, score in enumerate(scores):
            parent_id = self.child_to_parent[self.child_ids[child_idx]]
            if parent_id not in parent_best or score > parent_best[parent_id]:
                parent_best[parent_id] = score

        ranked = sorted(parent_best.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                "parent_id": parent_id,
                "text": self.parents[parent_id].page_content,
                "retrieval_score": score,
                "metadata": self.parents[parent_id].metadata,
            }
            for parent_id, score in ranked
        ]