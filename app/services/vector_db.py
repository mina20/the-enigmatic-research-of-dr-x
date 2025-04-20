import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
class SimpleVectorDB:
    def __init__(self, embedding_dim, model_name="all-MiniLM-L6-v2", device='cpu'):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []  # list of dicts aligned with index
        self.embedder = SentenceTransformer(model_name, device=device)

    def add(self, embeddings, metadata_list):
        """
        embeddings: np.array of shape (n_chunks, embedding_dim)
        metadata_list: list of dicts with chunk_text, file_name, etc.
        """
        assert len(embeddings) == len(metadata_list), "Mismatch between embeddings and metadata"
        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

    def search(self, query_text, top_k=5):
        """
        query_text: string query
        returns: top_k metadata dicts with similarity scores
        """
        query_embedding = self.embedder.encode([query_text])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(dist)
                results.append(result)
        return results
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)