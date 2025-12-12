from typing import Dict, Any
from .base_agent import Agent

ANALYST_SYSTEM_PROMPT = """
You are an analyst agent.

You receive:
- the clarified question
- a list of subtasks
- retrieved evidence snippets with doc_ids and scores.

Your job:
1. Synthesize a clear, structured answer to the question.
2. Explicitly reference evidence by doc_id where appropriate (e.g. "According to doc1_intro_to_multi_agent, ...").
3. If evidence seems weak or missing, explicitly state this in the answer.

Respond in plain text, but well structured with headings or bullet points if needed.
"""


class AnalystAgent(Agent):
    def __init__(self, llm_client):
        super().__init__("analyst", ANALYST_SYSTEM_PROMPT, llm_client)

    def run(self, clarified_question: str, evidence: Dict[str, Any]) -> str:
        # evidence is expected to match RetrieverAgent output
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Clarified question:\n{clarified_question}\n\n"
                    f"Evidence (JSON):\n{evidence}"
                )
            },
        ]
        answer = self.llm.chat(messages)
        return answer
