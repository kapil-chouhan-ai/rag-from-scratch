class Retriever:
    
    def __init__(self, embedder, vector_store, child_to_parent, parents):
        self.embedder = embedder
        self.vector_store = vector_store
        self.child_to_parent = child_to_parent
        self.parents = parents
#parent child retrieval strategy

    def retrieve(self, query, k = 10):
        q_emb = self.embedder.embed_query(query)
        D, I = self.vector_store.search(q_emb, k)

        result = []
        parent = []
        parent_dist = {}
        for idx , dist in zip(I[0], D[0]):
            parent_id = self.child_to_parent[idx]

            if parent_id in parent_dist:
                parent_dist[parent_id] = min(parent_dist[parent_id],dist)
            else :
                parent_dist[parent_id] = dist

        parent_dist = sorted(parent_dist.items(), key = lambda x: x[1])
        for parent_id, dist in parent_dist:
            result.append(
                {
                    'parent_id': parent_id,
                    'text':self.parents[parent_id].page_content,
                    'retrieval_score':dist,
                    'metadata':self.parents[parent_id].metadata
                }
            )

        return result