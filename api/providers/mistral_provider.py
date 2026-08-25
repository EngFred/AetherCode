from typing import Any, Dict, List, Optional
from openai import OpenAI
from providers.base_provider import BaseAIProvider
import config


class MistralExecutorProvider(BaseAIProvider):
    """
    Mistral AI Provider implementation.

    Acts as Stage 2 in the executor fallback chain:
        Groq → Mistral AI → OpenRouter → Gemini

    Uses Mistral's OpenAI-compatible endpoint with the free Experiment tier.
    Rate limit: 1 req/sec, 500k tokens/min (ongoing, never expires).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.MISTRAL_API_KEY
        self.model_name = model_name or config.MISTRAL_EXECUTOR_MODEL
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialise the OpenAI client pointing to Mistral API."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.mistral.ai/v1",
                timeout=config.PROVIDER_REQUEST_TIMEOUT_SECONDS,
            )
        return self._client

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
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
            raise RuntimeError(f"Mistral Provider Error: {str(e)}")
