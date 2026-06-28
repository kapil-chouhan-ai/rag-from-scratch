import numpy as np

class Embedder:

    def __init__(self, emb_model, query_instruction=""):
        self.model = emb_model
        self.query_instruction = query_instruction
 
    def embed_docs(self, texts):
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=True, batch_size=16
        )
 
    def embed_query(self, query):
        text = f"{self.query_instruction}{query}" if self.query_instruction else query
        return self.model.encode([text], convert_to_numpy=True).astype(np.float32)

#---------------------------------------------------------------------------------#
class BaseEmbeddingModel:

    def sentencetransformer(self):
        pass
    def BGE(self):
        pass
    def E5(self):
        pass
    def openaiembed(self):
        pass


