class RAGPipeline:

    def __init__(
        self,
        retriever,
        reranker,
        context_builder,
        generator
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator

    def run(self, query, retrieve_k=10, rerank_k=5):

        docs = self.retriever.retrieve(
            query=query,
            k=retrieve_k
        )
        docs = self.reranker.rerank(
            query=query,
            docs=docs,
            top_k=rerank_k
        )
        context = self.context_builder.build(
            docs
        )
        answer = self.generator.generate(
            query=query,
            context=context
        )

        return {
            "query": query,
            "documents": docs,
            "context": context,
            "answer": answer
        }