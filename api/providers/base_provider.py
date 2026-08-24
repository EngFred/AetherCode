from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseAIProvider(ABC):
    """
    Abstract Base Class defining the interface for AI Providers.
    Allows easy expansion or swapping of LLM services in the future.
    """

    @abstractmethod
    def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """Sends a prompt request to the AI model and returns the response."""
        pass