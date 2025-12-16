import uuid
from .base import BaseChunker

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, metadata=None):
        metadata = metadata or {}
        chunks = []

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": metadata
            })

            start = end - self.overlap

        return chunks
