from __future__ import annotations
import os
from openai import OpenAI
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class OpenAIClient(BaseLLMClient):
    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _generate_inner(self, prompt: str, **kwargs) -> str:
        # Handle parameter renaming for newer models (o1, etc.)
        if "max_tokens" in kwargs:
            # Models starting with 'o1' require 'max_completion_tokens'
            if self.model.startswith("o1-") or "gpt-5" in self.model:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content or ""
