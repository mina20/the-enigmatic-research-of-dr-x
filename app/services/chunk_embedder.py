# chunk_embedder.py

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np
import tiktoken
class ChunkEmbedder:
    def __init__(self, config):
        # self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # self.tokenizer = AutoTokenizer.from_pretrained.get_encoding(config.get("tokenizer_name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.chunk_size = config.get("chunk_size", 500)
        self.overlap = config.get("overlap", 50)
        model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        device = config.get("device", "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name, device=device)

        # Explicitly load model on CPU (though it defaults to CPU if no GPU available)
        # self.embedder = SentenceTransformer(model_name, device=device)

    def chunk_text(self, text):
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk = tokens[i:i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk))
        return chunks

    def embed_chunks(self, chunks):
        embeddings = self.embedder.encode(chunks)
        return embeddings

    def chunk_and_embed(self, all_data):
        processed_chunks = []
        all_embeddings = []

        for item in all_data:
            text = item["content"]
            if not text.strip():
                continue
            chunks = self.chunk_text(text)
            embeddings = self.embed_chunks(chunks)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                processed_chunks.append({
                    "file_name": item["file_name"],
                    "page_or_sheet": item["page_or_sheet"],
                    "chunk_number": i + 1,
                    "chunk_text": chunk
                })
                all_embeddings.append(emb)

        return processed_chunks, np.array(all_embeddings)