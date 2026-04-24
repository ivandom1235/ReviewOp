from __future__ import annotations
from abc import ABC, abstractmethod
from ..config import BuilderConfig
from .disk_cache import LLMDiskCache

class BaseLLMClient(ABC):
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg
        self.model = cfg.llm_model
        self.cache = LLMDiskCache() if getattr(cfg, "use_cache", True) else None

    def generate(self, prompt: str, **kwargs) -> str:
        if self.cache:
            cached = self.cache.get(prompt, self.model)
            if cached:
                return cached
        
        response = self._generate_inner(prompt, **kwargs)
        
        if self.cache:
            self.cache.set(prompt, self.model, response)
        return response

    @abstractmethod
    def _generate_inner(self, prompt: str, **kwargs) -> str:
        """Actual generation logic."""
        pass
