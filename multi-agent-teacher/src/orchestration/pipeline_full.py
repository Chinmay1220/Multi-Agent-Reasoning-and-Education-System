from typing import Dict, Any
from ..llm_client import LLMClient
from ..retrieval import SimpleEmbeddingRetriever
from ..agents.planner import PlannerAgent
from ..agents.retriever_agent import RetrieverAgent
from ..agents.analyst import AnalystAgent
from ..agents.critic import CriticAgent


class MultiAgentPipeline:
    """
    Full pipeline:
    Planner -> RetrieverAgent -> Analyst -> Critic
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.retriever_backend = SimpleEmbeddingRetriever(llm_client)

        self.planner = PlannerAgent(llm_client)
        self.retriever_agent = RetrieverAgent(llm_client, self.retriever_backend)
        self.analyst = AnalystAgent(llm_client)
        self.critic = CriticAgent(llm_client)

    def answer_question(self, question: str) -> Dict[str, Any]:
        plan = self.planner.run(question=question)

        clarified = plan.get("clarified_question", question)
        subtasks = plan.get("subtasks", [clarified])

        retrieval_result = self.retriever_agent.run(
            clarified_question=clarified,
            subtasks=subtasks
        )

        draft_answer = self.analyst.run(
            clarified_question=clarified,
            evidence=retrieval_result
        )

        critique = self.critic.run(
            clarified_question=clarified,
            evidence=retrieval_result,
            draft_answer=draft_answer
        )

        final_answer = critique.get("final_answer", draft_answer)

        return {
            "original_question": question,
            "plan": plan,
            "retrieval": retrieval_result,
            "draft_answer": draft_answer,
            "critique": critique,
            "final_answer": final_answer,
        }
