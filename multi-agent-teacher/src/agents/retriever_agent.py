from typing import Dict, Any, List
from .base_agent import Agent
from ..retrieval import SimpleEmbeddingRetriever


class RetrieverAgent(Agent):
    """
    Uses the retrieval backend to fetch evidence for each subtask.
    """

    def __init__(self, llm_client, retriever: SimpleEmbeddingRetriever):
        super().__init__("retriever", "You are a retrieval agent.", llm_client)
        self.retriever_backend = retriever

    def run(self, clarified_question: str, subtasks: List[str]) -> Dict[str, Any]:
        evidence = []
        for sub in subtasks:
            query = f"{clarified_question} | subtask: {sub}"
            snippets = self.retriever_backend.retrieve(query)
            evidence.append({
                "subtask": sub,
                "snippets": snippets
            })
        return {
            "clarified_question": clarified_question,
            "evidence": evidence
        }
