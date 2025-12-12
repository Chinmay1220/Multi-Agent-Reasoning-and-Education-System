from typing import Dict, Any
from ..llm_client import LLMClient
from ..retrieval import SimpleEmbeddingRetriever
from ..agents.analyst import AnalystAgent
from ..agents.critic import CriticAgent


class TwoAgentPipeline:
    """
    Baseline pipeline:
    - Retrieval is done implicitly by the Analyst (not ideal but simpler)
    - Critic checks the answer against retrieved context

    This is good for Notebook 2 teaching progression:
    Single agent -> two-agent (analyst + critic).
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.retriever_backend = SimpleEmbeddingRetriever(llm_client)
        self.analyst = AnalystAgent(llm_client)
        self.critic = CriticAgent(llm_client)

    def answer_question(self, question: str) -> Dict[str, Any]:
        # For simplicity: use the question directly to retrieve evidence
        evidence = self.retriever_backend.retrieve(question)
        evidence_payload = {
            "clarified_question": question,
            "evidence": [
                {
                    "subtask": "overall",
                    "snippets": evidence
                }
            ]
        }

        draft_answer = self.analyst.run(
            clarified_question=question,
            evidence=evidence_payload
        )

        critique = self.critic.run(
            clarified_question=question,
            evidence=evidence_payload,
            draft_answer=draft_answer
        )

        final_answer = critique.get("final_answer", draft_answer)

        return {
            "question": question,
            "evidence": evidence_payload,
            "draft_answer": draft_answer,
            "critique": critique,
            "final_answer": final_answer,
        }
