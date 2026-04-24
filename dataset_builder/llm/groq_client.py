from __future__ import annotations
import os
from openai import OpenAI
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class GroqClient(BaseLLMClient):
    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        # Groq is OpenAI-compatible
        self.client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

    def _generate_inner(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content or ""
