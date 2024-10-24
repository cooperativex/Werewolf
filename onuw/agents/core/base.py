from typing import Dict
from abc import abstractmethod

from ..roles import BaseRole
from ...backends import IntelligenceBackend


class AgentCore:
    """An abstraction of the agents."""
    def __init__(self, role: BaseRole, backend: IntelligenceBackend, global_prompt: str, **kwargs):
        self.backend = backend
        self.role = role
        self.name = self.role.name
        self.role_desc = self.role.role_description
        self.global_prompt = global_prompt
    
    @abstractmethod
    def act(self, observation: Dict):
        raise NotImplementedError
