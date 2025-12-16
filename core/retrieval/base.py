from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Returns retrieved chunks:
        {
          "chunk_id": str,
          "text": str,
          "score": float,
          "metadata": dict
        }
        """
        pass
