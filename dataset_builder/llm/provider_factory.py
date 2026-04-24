from __future__ import annotations
from ..config import BuilderConfig
from .base_client import BaseLLMClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .groq_client import GroqClient

def get_llm_client(cfg: BuilderConfig) -> BaseLLMClient:
    """Factory method to get the appropriate LLM client."""
    provider = str(cfg.llm_provider).lower()
    if provider == "openai":
        return OpenAIClient(cfg)
    elif provider == "anthropic":
        return AnthropicClient(cfg)
    elif provider == "groq":
        return GroqClient(cfg)
    elif provider == "none":
        # We could return a MockClient or just raise
        raise ValueError("LLM provider is set to 'none'")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
