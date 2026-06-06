class DocumentLoader:

    def __init__(self, loader_cls):
        self.loader_cls = loader_cls
        
    def load(self, path):
        loader = self.loader_cls(path)
        return loader.load()