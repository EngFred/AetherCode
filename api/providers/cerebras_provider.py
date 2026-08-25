from typing import Any, Dict, List, Optional
from openai import OpenAI
from providers.base_provider import BaseAIProvider
import config


class CerebrasExecutorProvider(BaseAIProvider):
    """
    Cerebras Provider implementation.

    Acts as the second-stage executor in the fallback chain:
        Groq → Cerebras → Gemini

    Uses the standard OpenAI Python SDK pointed at Cerebras' inference
    endpoint — 100 % OpenAI-compatible, so it plugs directly into
    run_groq_tool_loop (aliased as run_cerebras_tool_loop in
    core/cerebras_tool_loop.py) without any special adaptation.

    Free tier: 1,000,000 tokens / day, resets daily.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.CEREBRAS_API_KEY
        self.model_name = model_name or config.CEREBRAS_EXECUTOR_MODEL
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialise the OpenAI client once and cache it."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.cerebras.ai/v1",
                timeout=config.CEREBRAS_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Sends a prompt to Cerebras and returns the raw response message.
        Signature mirrors GroqExecutorProvider.generate_response so both
        providers are interchangeable at the call site.
        """
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            raise RuntimeError(f"Cerebras Provider Error: {str(e)}")
