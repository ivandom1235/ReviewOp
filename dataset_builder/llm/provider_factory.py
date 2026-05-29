from __future__ import annotations
from ..config import BuilderConfig
from .base_client import BaseLLMClient

def get_llm_client(cfg: BuilderConfig, wrap_fallback: bool = True) -> BaseLLMClient:
    """Factory method to get the appropriate LLM client."""
    provider = str(cfg.llm_provider).lower()
    client = None
    
    if provider == "openai":
        from .openai_client import OpenAIClient
        client = OpenAIClient(cfg)
    elif provider in ("anthropic", "claude"):
        from .anthropic_client import AnthropicClient
        client = AnthropicClient(cfg)
    elif provider == "gemini":
        from .gemini_client import GeminiClient
        client = GeminiClient(cfg)
    elif provider == "groq":
        from .groq_client import GroqClient
        client = GroqClient(cfg)
    elif provider == "openrouter":
        from .openrouter_client import OpenRouterClient
        client = OpenRouterClient(cfg)
    elif provider == "huggingface":
        from .huggingface_client import HuggingFaceClient
        client = HuggingFaceClient(cfg)
    elif provider == "ollama":
        from .ollama_client import OllamaClient
        client = OllamaClient(cfg)
    elif provider == "lightning":
        from .lightning_client import LightningClient
        client = LightningClient(cfg)
    elif provider == "none":
        raise ValueError("LLM provider is set to 'none'")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if wrap_fallback and provider != "none":
        from .fallback_client import FallbackLLMClient
        return FallbackLLMClient(cfg, client)
    
    return client
