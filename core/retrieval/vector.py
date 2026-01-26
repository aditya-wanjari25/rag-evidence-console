import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI

from .base import BaseRetriever

class OpenAIFaissRetriever(BaseRetriever):
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.embedding_model = embedding_model
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []

    def _embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [x.embedding for x in resp.data]
        return np.array(vectors, dtype="float32")

    def add_chunks(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None or len(self.chunks) == 0:
            raise RuntimeError("Retriever index is empty. Run ingest first (or load an existing index).")

        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })
        return results

    def save(self, dir_path: str):
        """
        Persists:
        - FAISS index to artifacts/faiss.index
        - chunks to artifacts/chunks.jsonl
        - manifest metadata to artifacts/manifest.json
        """
        if self.index is None:
            raise RuntimeError("No FAISS index to save.")

        os.makedirs(dir_path, exist_ok=True)

        index_path = os.path.join(dir_path, "faiss.index")
        chunks_path = os.path.join(dir_path, "chunks.jsonl")
        manifest_path = os.path.join(dir_path, "manifest.json")

        faiss.write_index(self.index, index_path)

        with open(chunks_path, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        manifest = {
            "embedding_model": self.embedding_model,
            "num_chunks": len(self.chunks)
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def load(self, dir_path: str):
        index_path = os.path.join(dir_path, "faiss.index")
        chunks_path = os.path.join(dir_path, "chunks.jsonl")
        manifest_path = os.path.join(dir_path, "manifest.json")

        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            raise FileNotFoundError(f"Missing index artifacts in: {dir_path}")

        self.index = faiss.read_index(index_path)

        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # Optional: load manifest info (and warn if embedding model differs)
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            saved_model = manifest.get("embedding_model")
            if saved_model and saved_model != self.embedding_model:
                # Not fatal, but important to know
                print(f"[warn] Embedding model differs. saved={saved_model}, current={self.embedding_model}")
