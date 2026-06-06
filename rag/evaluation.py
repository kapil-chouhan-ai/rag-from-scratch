class RetrievalEvaluator:

    def __init__(self, retriever):
        self.retriever = retriever

    def hit_rate(self, dataset, k=5):

        hits = 0

        for sample in dataset:

            query = sample["question"]
            expected_page = str(sample["expected_page"]) # Convert to string for comparison

            docs = self.retriever.retrieve(
                query,
                k=k
            )

            pages = [
                doc["metadata"]["page_num"]
                for doc in docs
            ]

            if expected_page in pages:
                hits += 1

        return hits / len(dataset)

    def recall_at_k(self, dataset, k=5):

        hits = 0

        total = len(dataset)

        for sample in dataset:

            query = sample["question"]
            expected_page = str(sample["expected_page"]) # Convert to string for comparison

            docs = self.retriever.retrieve(
                query,
                k=k
            )

            pages = [
                doc["metadata"]["page_num"]
                for doc in docs
            ]

            if expected_page in pages:
                hits += 1

        return hits / total

    def evaluate(self, dataset, k=5):

        return {
            "Hit@K": self.hit_rate(
                dataset,
                k
            ),
            "Recall@K": self.recall_at_k(
                dataset,
                k
            )
        }