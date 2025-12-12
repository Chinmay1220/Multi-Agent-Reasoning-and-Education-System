from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import numpy as np

from .config import CORPUS_DIR, TOP_K_SNIPPETS
from .llm_client import LLMClient


@dataclass
class DocumentChunk:
    doc_id: str
    text: str
    embedding: np.ndarray


class SimpleEmbeddingRetriever:
    """
    Minimal retrieval system:
    - loads .txt docs from corpus folder
    - keeps each file as a single chunk (for simplicity)
    - embeds and stores them in memory
    - retrieves top-k by cosine similarity
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.chunks: List[DocumentChunk] = []
        self._loaded = False

    def load_corpus(self, corpus_dir: Path = CORPUS_DIR):
        texts = []
        doc_ids = []

        for path in corpus_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            texts.append(text)
            doc_ids.append(path.name)

        if not texts:
            raise ValueError(f"No .txt files found in corpus dir: {corpus_dir}")

        embeddings = self.llm.embed_texts(texts)
        self.chunks = [
            DocumentChunk(doc_id=doc_id, text=text, embedding=np.array(emb))
            for doc_id, text, emb in zip(doc_ids, texts, embeddings)
        ]
        self._loaded = True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def retrieve(self, query: str, k: int = TOP_K_SNIPPETS) -> List[Dict]:
        if not self._loaded:
            self.load_corpus()

        query_emb = np.array(self.llm.embed_texts([query])[0])

        scored = []
        for chunk in self.chunks:
            score = self._cosine_similarity(query_emb, chunk.embedding)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        return [
            {
                "doc_id": chunk.doc_id,
                "score": score,
                "text": chunk.text,
            }
            for score, chunk in top
        ]
