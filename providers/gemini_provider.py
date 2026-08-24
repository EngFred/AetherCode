from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from providers.base_provider import BaseAIProvider
import config

class GeminiAnalyzerProvider(BaseAIProvider):
    """
    Gemini Provider implementation.
    Acts as 'The Detective' to scan directory maps and analyze project structures.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model_name = model_name or config.GEMINI_ANALYZER_MODEL
        self._client = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_response(
        self, 
        prompt: str, 
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Sends the file map and prompt to Gemini to pinpoint target bug files.
        """
        try:
            gen_config = None
            if system_instruction:
                gen_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=gen_config
            )
            return response.text or ""
        except Exception as e:
            return f"Gemini Provider Error: {str(e)}"