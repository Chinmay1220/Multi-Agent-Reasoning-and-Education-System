from typing import List, Dict, Any
from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL, EMBEDDING_MODEL

class LLMClient:
    """
    Thin wrapper around the OpenAI client.
    Provides:
    - chat() for chat completions
    - embed_texts() for embeddings
    """

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Please configure your .env.")
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Returns list of embedding vectors (one per text).
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
