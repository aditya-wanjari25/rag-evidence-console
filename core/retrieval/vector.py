import faiss
import numpy as np
from typing import List, Dict
from openai import OpenAI

from .base import BaseRetriever

class OpenAIFaissRetriever(BaseRetriever):
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.embedding_model = embedding_model

        self.index = None
        self.chunks = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        embeddings = [e.embedding for e in response.data]
        return np.array(embeddings).astype("float32")

    def add_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self._embed([query])
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })

        return results
