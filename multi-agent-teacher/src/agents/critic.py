import json
from typing import Dict, Any
from .base_agent import Agent

CRITIC_SYSTEM_PROMPT = """
You are a critic agent and computational skeptic.

You receive:
- the clarified question
- the retrieved evidence
- the analyst's draft answer

Your tasks:
1. Check whether the answer is supported by the evidence.
2. Identify any hallucinations (claims not explicitly supported by the evidence).
3. Suggest improvements if needed.

Respond ONLY with a JSON object of the form:

{
  "supported": true or false,
  "issues": ["...","..."],
  "suggested_revision": "revised answer or empty string if not needed",
  "final_answer": "the answer that should be shown to the user"
}

If the answer is mostly correct but needs small edits, incorporate them in final_answer.
If the answer is badly unsupported, explain in issues and either leave final_answer empty or provide a conservative alternative.
"""


class CriticAgent(Agent):
    def __init__(self, llm_client):
        super().__init__("critic", CRITIC_SYSTEM_PROMPT, llm_client)

    def run(
        self,
        clarified_question: str,
        evidence: Dict[str, Any],
        draft_answer: str
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Clarified question:\n{clarified_question}\n\n"
                    f"Evidence (JSON):\n{json.dumps(evidence, indent=2)}\n\n"
                    f"Draft answer:\n{draft_answer}"
                ),
            },
        ]
        raw = self.llm.chat(messages)
        try:
            critique = json.loads(raw)
        except json.JSONDecodeError:
            critique = {
                "supported": False,
                "issues": ["Critic response was not valid JSON."],
                "suggested_revision": "",
                "final_answer": draft_answer,
                "raw_response": raw,
                "parse_error": True,
            }
        return critique
