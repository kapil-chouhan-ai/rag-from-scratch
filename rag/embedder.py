import numpy as np

class Embedder:

    def __init__(self, emb_model):
        self.model = emb_model

    def embed_docs(self, texts):
        emb = self.model.encode(texts, convert_to_numpy = True, show_progress_bar = True, batch_size = 16)
        return emb

    def embed_query(self, query):
        emb = self.model.encode([query], convert_to_numpy = True).astype(np.float32)
        return emb

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