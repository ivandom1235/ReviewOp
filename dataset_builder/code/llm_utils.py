from __future__ import annotations
import os
import json
import requests
import hashlib
import threading
import asyncio
import httpx
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

class LlmProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        pass

class RunPodProvider(LlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.base_url = base_url or os.environ.get("RUNPOD_ENDPOINT_URL")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        if not self.api_key or not self.base_url:
            raise ValueError("RunPod API key and Endpoint URL are required")
        data = {
            "input": {
                "prompt": prompt,
                "model_name": model_name,
                **kwargs
            }
        }
        response = self.session.post(self.base_url, json=data)
        response.raise_for_status()
        result = response.json()
        if "output" in result:
            return str(result["output"])
        return json.dumps(result)

class OpenAiProvider(LlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        res = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

class OllamaProvider(LlmProvider):
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or "http://localhost:11434"
        self.session = requests.Session()

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        res = self.session.post(f"{self.base_url}/api/generate", json=payload)
        res.raise_for_status()
        return res.json()["response"]

class MockProvider(LlmProvider):
    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        return "Mock LLM Response"

class AsyncLlmProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        pass

class AsyncRunPodProvider(AsyncLlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.base_url = base_url or os.environ.get("RUNPOD_ENDPOINT_URL")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if self.api_key else {}

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        if not self.api_key or not self.base_url:
            raise ValueError("RunPod API key and Endpoint URL are required")
        data = {
            "input": {
                "prompt": prompt,
                "model_name": model_name,
                **kwargs
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(self.base_url, json=data, headers=self.headers)
            res.raise_for_status()
            result = res.json()
            if "output" in result:
                return str(result["output"])
            return json.dumps(result)

class AsyncOpenAiProvider(AsyncLlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

class AsyncOllamaProvider(AsyncLlmProvider):
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or "http://localhost:11434"

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/api/generate", json=payload)
            res.raise_for_status()
            return res.json()["response"]

class AsyncMockProvider(AsyncLlmProvider):
    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        await asyncio.sleep(0.01)
        return "Async Mock LLM Response"

class PersistentCache:
    def __init__(self, cache_file: str = ".llm_cache.json"):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        self._data: Dict[str, str] = {}
        self._dirty = False
        self._load()
        import atexit
        atexit.register(self.flush)

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def get(self, key: str) -> Optional[str]:
        # Lock-free fast path for reads (Python dict access is thread-safe due to the GIL)
        return self._data.get(key)

    def set(self, key: str, value: str):
        with self.lock:
            if self._data.get(key) == value:
                return
            self._data[key] = value
            self._dirty = True

    def clear(self):
        with self.lock:
            self._data.clear()
            self._dirty = True

    def flush(self):
        with self.lock:
            if not self._dirty:
                return
            try:
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                self._dirty = False
            except IOError:
                pass


GLOBAL_LLM_CACHE = PersistentCache()


def flush_llm_cache() -> None:
    GLOBAL_LLM_CACHE.flush()


def get_provider(provider_type: str, api_key: str | None = None, base_url: str | None = None) -> LlmProvider:
    provider_key = str(provider_type or "").strip().lower()
    if provider_key == "runpod":
        return RunPodProvider(api_key, base_url)
    elif provider_key == "openai":
        return OpenAiProvider(api_key, base_url)
    elif provider_key == "ollama":
        return OllamaProvider(base_url)
    elif provider_key == "mock":
        return MockProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_type}")


def get_async_provider(provider_type: str, api_key: str | None = None, base_url: str | None = None) -> AsyncLlmProvider:
    provider_key = str(provider_type or "").strip().lower()
    if provider_key == "runpod":
        return AsyncRunPodProvider(api_key, base_url)
    elif provider_key == "openai":
        return AsyncOpenAiProvider(api_key, base_url)
    elif provider_key == "ollama":
        return AsyncOllamaProvider(base_url)
    elif provider_key == "mock":
        return AsyncMockProvider()
    else:
        raise ValueError(f"Unsupported async provider: {provider_type}")


def resolve_llm_provider(provider_name: str | None, model_name: str | None = None) -> LlmProvider | None:
    if not provider_name:
        return None
    provider_key = str(provider_name).strip().lower()
    if provider_key in {"", "none"}:
        return None
    return get_provider(provider_key)


def resolve_async_llm_provider(provider_name: str | None, model_name: str | None = None) -> AsyncLlmProvider | None:
    if not provider_name:
        return None
    provider_key = str(provider_name).strip().lower()
    if provider_key in {"", "none"}:
        return None
    return get_async_provider(provider_key)


def discover_best_provider() -> Tuple[str, Optional[str], Optional[str]]:
    runpod_key = os.environ.get("RUNPOD_API_KEY")
    runpod_url = os.environ.get("RUNPOD_ENDPOINT_URL")
    if runpod_key and runpod_url:
        return "runpod", runpod_key, runpod_url
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return "openai", openai_key, None
    
    return "ollama", None, os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def reason_implicit_signal(text: str, candidate_aspects: List[str], provider: LlmProvider, model_name: str) -> List[str]:
    prompt = f"""Given the following review segment: "{text}"
Rephrase it such that any implicit aspects from this list: {candidate_aspects} become explicit.
If the segment does not imply any of these aspects, return the original text.
Return ONLY the rephrased text.
Review segment: {text}
Explicit version:"""
    
    cache_key = hashlib.md5(f"{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key)
    if cached_val:
        return [cached_val]

    try:
        paraphrase = provider.generate(prompt, model_name, temperature=0.2)
        result = paraphrase.strip()
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return [result]
    except Exception:
        return [text]


async def reason_implicit_signal_async(text: str, candidate_aspects: List[str], provider: AsyncLlmProvider, model_name: str) -> List[str]:
    prompt = f"""Given the following review segment: "{text}"
Rephrase it such that any implicit aspects from this list: {candidate_aspects} become explicit.
If the segment does not imply any of these aspects, return the original text.
Return ONLY the rephrased text.
Review segment: {text}
Explicit version:"""
    
    cache_key = hashlib.md5(f"async:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key)
    if cached_val:
        return [cached_val]

    try:
        paraphrase = await provider.generate(prompt, model_name, temperature=0.2)
        result = paraphrase.strip()
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return [result]
    except Exception:
        return [text]


def augment_implicit_difficulty(text: str, aspect: str, provider: LlmProvider, model_name: str, domain: str = "general") -> str:
    prompt = f"""Task: Adversarial Implicit Aspect Rephrasing (Research-Grade)
Domain: {domain}
Target Aspect: {aspect}
Original Segment: "{text}"

Goal: Rewrite the segment to be HIGH-DIFFICULTY IMPLICIT.
Rules:
1. DO NOT use the word "{aspect}" or its direct synonyms.
2. DO NOT use surface keywords typically mapped to this aspect (e.g., if aspect is 'price', avoid 'cheap', 'expensive', 'cost').
3. Focus on symptoms, consequences, or experiential details (e.g., instead of "short battery life", use "I couldn't even finish my morning commute before it went dark").
4. Maintain the original sentiment.
5. Keep it concise and human-like.

Return ONLY the rewritten hard-implicit segment.
Hard-Implicit Version:"""
    
    cache_key = hashlib.md5(f"hard:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key)
    if cached_val:
        return cached_val

    try:
        hard_ver = provider.generate(prompt, model_name, temperature=0.7)
        result = hard_ver.strip().strip('"')
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return result
    except Exception:
        return ""


async def augment_implicit_difficulty_async(text: str, aspect: str, provider: AsyncLlmProvider, model_name: str, domain: str = "general") -> str:
    prompt = f"""Task: Adversarial Implicit Aspect Rephrasing (Research-Grade)
Domain: {domain}
Target Aspect: {aspect}
Original Segment: "{text}"

Goal: Rewrite the segment to be HIGH-DIFFICULTY IMPLICIT.
Rules:
1. DO NOT use the word "{aspect}" or its direct synonyms.
2. DO NOT use surface keywords typically mapped to this aspect (e.g., if aspect is 'price', avoid 'cheap', 'expensive', 'cost').
3. Focus on symptoms, consequences, or experiential details (e.g., instead of "short battery life", use "I couldn't even finish my morning commute before it went dark").
4. Maintain the original sentiment.
5. Keep it concise and human-like.

Return ONLY the rewritten hard-implicit segment.
Hard-Implicit Version:"""
    
    cache_key = hashlib.md5(f"async-hard:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key)
    if cached_val:
        return cached_val

    try:
        hard_ver = await provider.generate(prompt, model_name, temperature=0.7)
        result = hard_ver.strip().strip('"')
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return result
    except Exception:
        return ""


def generate_multi_aspect_review(aspects: List[str], provider: LlmProvider, model_name: str, domain: str = "general") -> str:
    aspects_str = ", ".join(aspects)
    prompt = f"""Task: Multi-Aspect Implicit Review Generation
Domain: {domain}
Target Aspects: {aspects_str}

Goal: Generate a single, natural-sounding review sentence or segment that implicitly suggests ALL listed aspects without explicitly naming them.
Avoid surface word leakage. Focus on consequences or interconnected symptoms.

Return ONLY the generated segment.
Multi-Aspect Implicit Segment:"""
    
    cache_key = hashlib.md5(f"multi:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key)
    if cached_val:
        return cached_val

    try:
        multi_ver = provider.generate(prompt, model_name, temperature=0.7)
        result = multi_ver.strip().strip('"')
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return result
    except Exception:
        return ""
