from abc import ABC, abstractmethod
from typing import List, Dict

class BaseChunker(ABC):
    """
    Responsible for splitting raw text into chunks
    """

    @abstractmethod
    def chunk(self, text: str, metadata: Dict | None = None) -> List[Dict]:
        """
        Returns a list of chunks with structure:
        {
          "chunk_id": str,
          "text": str,
          "metadata": dict
        }
        """
        pass
