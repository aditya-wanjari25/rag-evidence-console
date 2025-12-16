from abc import ABC, abstractmethod
from typing import List, Dict

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[Dict]) -> Dict:
        """
        Returns:
        {
          "answer": str,
          "metadata": dict
        }
        """
        pass
