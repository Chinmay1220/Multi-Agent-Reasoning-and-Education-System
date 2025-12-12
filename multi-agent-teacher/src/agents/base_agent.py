from abc import ABC, abstractmethod
from typing import Any, Dict
from ..llm_client import LLMClient


class Agent(ABC):
    """
    Base class for all agents.
    Each agent:
    - has a name
    - has a system prompt
    - uses the shared LLM client
    """

    def __init__(self, name: str, system_prompt: str, llm_client: LLMClient):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm_client

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Subclasses implement this to perform their role.
        """
        pass
