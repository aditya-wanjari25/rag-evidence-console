from typing import List, Dict
from openai import OpenAI

from .base import BaseGenerator

class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0 ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def generate(self, query: str, context: List[Dict]) -> Dict:
        context_text = "\n\n".join(
            f"[Source {i}] {c['text']}"
            for i, c in enumerate(context)
        )

        prompt = f"""
        Use the context below to answer the question.
        If the answer is not contained in the context, say so clearly.

        Context:
        {context_text}

        Question:
        {query}
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature= self.temperature
        )

        return {
            "answer": response.choices[0].message.content,
            "metadata": {
                "model": self.model,
                "num_sources": len(context)
            }
        }
