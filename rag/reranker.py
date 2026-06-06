class Reranker:
    
    def __init__(self, model = None):
        self.model = model

    def rerank(self, query, docs, top_k = 5):
        if self.model is None:
            return docs[:top_k] if top_k else docs

        else:
            pairs = [
                (query, doc['text']) for doc in docs
            ]
            scores = self.model.predict(pairs)
            for score, doc in zip(scores, docs):
                doc['rerank_score'] = score

            docs = sorted(docs, key = lambda x: x['rerank_score'], reverse= True)
            return docs[:top_k]