import json
from typing import Dict, Any
from .base_agent import Agent

PLANNER_SYSTEM_PROMPT = """
You are a planning agent in a multi-agent LLM system.

Your job:
1. Clarify the user's question if needed.
2. Break it into 2-5 concrete, answerable sub-tasks.

Respond ONLY with a valid JSON object:

{
  "clarified_question": "...",
  "subtasks": [
    "subtask 1",
    "subtask 2"
  ]
}

Do not include any additional commentary outside the JSON.
"""


class PlannerAgent(Agent):
    def __init__(self, llm_client):
        super().__init__("planner", PLANNER_SYSTEM_PROMPT, llm_client)

    def run(self, question: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        raw = self.llm.chat(messages)
        # Try to parse JSON; if it fails, wrap it.
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {
                "clarified_question": question,
                "subtasks": [question],
                "raw_response": raw,
                "parse_error": True,
            }
        return plan
