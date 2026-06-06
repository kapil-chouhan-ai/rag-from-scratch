from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    
    def __init__(
        self,
        child_chunk_size,
        child_chunk_overlap,
        parent_chunk_size,
        parent_chunk_overlap
    ):
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )

#----------------------parent-child implimentation--------------------------

    def split(self, documents):
        parent_chunks = self.parent_splitter.split_documents(documents)
        parents = {}
        children = {}
        child_to_parent = {}
        parent_id, child_id = 0, 0

        for parent in parent_chunks:
            parents[parent_id] = parent
            child_doc = self.child_splitter.split_documents([parent])
            
            for child in child_doc:
                children[child_id] = child
                child_to_parent[child_id] = parent_id
                child_id += 1

            parent_id += 1

        return parents, children, child_to_parent