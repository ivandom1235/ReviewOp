from __future__ import annotations
import os
from anthropic import Anthropic
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class AnthropicClient(BaseLLMClient):
    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _generate_inner(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude 3 returns a list of content blocks
        return response.content[0].text if response.content else ""
