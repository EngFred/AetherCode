from typing import Any, Dict, List, Optional
from openai import OpenAI
from providers.base_provider import BaseAIProvider
import config


class OpenRouterExecutorProvider(BaseAIProvider):
    """
    OpenRouter Provider implementation.

    Acts as Stage 3 in the executor fallback chain:
        Groq → Mistral AI → OpenRouter → Gemini

    Uses OpenRouter's free-tier models (with ':free' suffix or 'openrouter/free')
    which have a 50 req/day quota that resets every 24 hours (no credit card required).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model_name = model_name or config.OPENROUTER_EXECUTOR_MODEL
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy-initialise the OpenAI client pointing to OpenRouter."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=config.PROVIDER_REQUEST_TIMEOUT_SECONDS,
                default_headers={
                    "HTTP-Referer": "https://github.com/EngFred/AetherCode",
                    "X-Title": "AetherCode",
                },
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
            raise RuntimeError(f"OpenRouter Provider Error: {str(e)}")
