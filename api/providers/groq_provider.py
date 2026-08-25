from typing import Any, Dict, List, Optional
from groq import Groq
from providers.base_provider import BaseAIProvider
import config

class GroqExecutorProvider(BaseAIProvider):
    """
    Groq Provider implementation.
    Acts as 'The Worker' to run function tools and execute local code edits.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model_name = model_name or config.GROQ_EXECUTOR_MODEL
        self._client = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            # timeout is enforced by the underlying httpx client for every
            # request made through this Groq instance, so both the tool-loop
            # calls and the general-chat calls in agent.py get it for free.
            self._client = Groq(api_key=self.api_key, timeout=config.GROQ_REQUEST_TIMEOUT_SECONDS)
        return self._client

    def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Any:
        """
        Sends prompts to Groq and handles function calling requests.
        """
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            raise RuntimeError(f"Groq Provider Error: {str(e)}")