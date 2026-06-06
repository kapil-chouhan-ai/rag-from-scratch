class ContextBuilder:

    def build(self, docs):
        context_parts = []
        for i, doc in enumerate(docs, start = 1):
            context_parts.append(f"Source{i}:from {doc['metadata']['source']} \npage = {doc['metadata']['page']}\n{doc['text']}")
        context = f"\n--------------------------\n".join(context_parts)
        return context

        