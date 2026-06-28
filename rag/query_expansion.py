class QueryExpander:
    """
    Generates paraphrased variants of the input query using the same LLM
    used for answer generation. Retrieval then runs against the original
    query plus each variant, and results are merged - this recovers
    documents that are relevant but phrased with different vocabulary
    than the literal query (the classic vocabulary-mismatch problem in
    lexical/dense retrieval).
    """

    def __init__(self, llm, num_variants=2):
        self.llm = llm
        self.num_variants = num_variants

    def expand(self, query):
        prompt = (
            "<|im_start|>system\n"
            f"Rewrite the user question into {self.num_variants} different "
            "but semantically equivalent questions. Output exactly "
            f"{self.num_variants} lines, no numbering, no extra text.<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        output = self.llm(prompt, max_new_tokens=80)[0]["generated_text"]
        output = output[len(prompt):]

        variants = [
            line.strip("-• ").strip()
            for line in output.strip().split("\n")
            if line.strip()
        ][: self.num_variants]

        return [query] + variants


def expand_and_retrieve(query, retriever, expander, k=10, rrf_k=60):
    """
    Runs `retriever.retrieve` for the original query and every expansion,
    then fuses the per-query ranked lists with the same RRF logic used
    by HybridRetriever (kept as a free function here rather than coupling
    QueryExpander to HybridRetriever's class, since expansion is
    orthogonal to dense-vs-hybrid retrieval and should compose with
    either).
    """
    queries = expander.expand(query) if expander else [query]

    fused_scores = {}
    doc_lookup = {}

    for q in queries:
        for rank, doc in enumerate(retriever.retrieve(q, k=k)):
            pid = doc["parent_id"]
            fused_scores[pid] = fused_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_lookup.setdefault(pid, doc)

    ranked_ids = sorted(fused_scores, key=lambda pid: fused_scores[pid], reverse=True)[:k]

    results = []
    for pid in ranked_ids:
        doc = dict(doc_lookup[pid])
        doc["retrieval_score"] = fused_scores[pid]
        results.append(doc)
    return results