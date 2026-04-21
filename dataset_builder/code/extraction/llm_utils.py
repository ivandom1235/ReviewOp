from __future__ import annotations
import os
import json
import requests
import hashlib
import threading
import asyncio
import httpx
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default

class LlmProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        pass

class RunPodProvider(LlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or _env_value("REVIEWOP_RUNPOD_API_KEY", "RUNPOD_API_KEY")
        self.base_url = base_url or _env_value("REVIEWOP_RUNPOD_ENDPOINT_URL", "RUNPOD_ENDPOINT_URL")
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
                **kwargs
            }
        }
        if model_name:
            data["input"]["model_name"] = model_name

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
        kwargs.pop("bypass_cache", None)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        res = self.session.post(url, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

class ClaudeProvider(LlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or _env_value("CLAUDE_API_KEY")
        self.base_url = base_url or _env_value("CLAUDE_BASE_URL", default="https://api.anthropic.com/v1")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(
                {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
            )

    def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        kwargs.pop("bypass_cache", None)
        payload = {
            "model": model_name,
            "max_tokens": int(kwargs.pop("max_tokens", 1024)),
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        url = self.base_url.rstrip("/") + "/messages"
        res = self.session.post(url, json=payload)
        res.raise_for_status()
        content = res.json().get("content", [])
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        return str(content)

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
        self.api_key = api_key or _env_value("REVIEWOP_RUNPOD_API_KEY", "RUNPOD_API_KEY")
        self.base_url = base_url or _env_value("REVIEWOP_RUNPOD_ENDPOINT_URL", "RUNPOD_ENDPOINT_URL")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        } if self.api_key else {}
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        # Late-bind the endpoint and key to ensure .env updates reflect in long-running processes
        self.api_key = self.api_key or _env_value("REVIEWOP_RUNPOD_API_KEY", "RUNPOD_API_KEY")
        self.base_url = self.base_url or _env_value("REVIEWOP_RUNPOD_ENDPOINT_URL", "RUNPOD_ENDPOINT_URL")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        } if self.api_key else {}
        
        if not self.api_key or not self.base_url:
            raise ValueError("RunPod API key and Endpoint URL are required")

        bypass = kwargs.pop("bypass_cache", False)
        data = {
            "input": {
                "prompt": prompt,
                **kwargs
            }
        }
        if model_name:
            data["input"]["model_name"] = model_name

        cache_key = hashlib.md5(f"runpod:{model_name}:{prompt}".encode("utf-8")).hexdigest()
        cached_val = GLOBAL_LLM_CACHE.get(cache_key, bypass=bypass)
        if cached_val:
            return cached_val


        # Ensure we use the synchronous endpoint for immediate response
        base_url = self.base_url
        if base_url.endswith("/run"):
            base_url = base_url.replace("/run", "/runsync")
        elif not base_url.endswith("/runsync"):
            base_url = base_url.rstrip("/") + "/runsync"

        client = self._get_client()
        res = await client.post(base_url, json=data, headers=self.headers)
        res.raise_for_status()
        result = res.json()
        # RunPod Serverless standard return is {"output": ..., "id": ..., "status": ...}
        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            # vLLM/Flash usually returns a string or a dict depending on the template
            output_text = output if isinstance(output, str) else json.dumps(output)
            GLOBAL_LLM_CACHE.set(cache_key, output_text)
            return output_text
        output_text = json.dumps(result)
        GLOBAL_LLM_CACHE.set(cache_key, output_text)
        return output_text

class AsyncOpenAiProvider(AsyncLlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        bypass = kwargs.pop("bypass_cache", False)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        
        cache_key = hashlib.md5(f"openai:{model_name}:{prompt}".encode("utf-8")).hexdigest()
        cached_val = GLOBAL_LLM_CACHE.get(cache_key, bypass=bypass)
        if cached_val:
            return cached_val

        url = self.base_url.rstrip("/") + "/chat/completions"
        client = self._get_client()
        res = await client.post(url, json=payload, headers=self.headers)
        res.raise_for_status()
        result = res.json()["choices"][0]["message"]["content"]
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return result

class AsyncClaudeProvider(AsyncLlmProvider):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or _env_value("CLAUDE_API_KEY")
        self.base_url = base_url or _env_value("CLAUDE_BASE_URL", default="https://api.anthropic.com/v1")
        self.headers = (
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            if self.api_key
            else {}
        )
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        bypass = kwargs.pop("bypass_cache", False)
        payload = {
            "model": model_name,
            "max_tokens": int(kwargs.pop("max_tokens", 1024)),
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        
        cache_key = hashlib.md5(f"claude:{model_name}:{prompt}".encode("utf-8")).hexdigest()
        cached_val = GLOBAL_LLM_CACHE.get(cache_key, bypass=bypass)
        if cached_val:
            return cached_val

        url = self.base_url.rstrip("/") + "/messages"
        client = self._get_client()
        res = await client.post(url, json=payload, headers=self.headers)
        res.raise_for_status()
        content = res.json().get("content", [])
        if isinstance(content, list):
            result = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            GLOBAL_LLM_CACHE.set(cache_key, result)
            return result
        result = str(content)
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return result

class AsyncOllamaProvider(AsyncLlmProvider):
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or "http://localhost:11434"
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def generate(self, prompt: str, model_name: str, **kwargs) -> str:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        client = self._get_client()
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

    def get(self, key: str, bypass: bool = False) -> Optional[str]:
        if bypass:
            return None
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


_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
GLOBAL_LLM_CACHE = PersistentCache(cache_file=str(_REPO_ROOT / ".llm_cache.json"))


def flush_llm_cache() -> None:
    GLOBAL_LLM_CACHE.flush()


def get_provider(provider_type: str, api_key: str | None = None, base_url: str | None = None) -> LlmProvider:
    provider_key = str(provider_type or "").strip().lower()
    if provider_key == "runpod":
        return RunPodProvider(api_key, base_url)
    elif provider_key == "openai":
        return OpenAiProvider(api_key, base_url)
    elif provider_key in {"claude", "anthropic"}:
        return ClaudeProvider(api_key, base_url)
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
    elif provider_key in {"claude", "anthropic"}:
        return AsyncClaudeProvider(api_key, base_url)
    elif provider_key == "ollama":
        return AsyncOllamaProvider(base_url)
    elif provider_key == "mock":
        return AsyncMockProvider()
    else:
        raise ValueError(f"Unsupported async provider: {provider_type}")


def resolve_async_llm_provider(provider_name: str | None, model_name: str | None = None) -> AsyncLlmProvider | None:
    if not provider_name:
        return None
    provider_key = str(provider_name).strip().lower()
    if provider_key in {"", "none"}:
        return None
    return get_async_provider(provider_key)


def resolve_processor_async_provider(processor_name: str | None, model_name: str | None = None) -> AsyncLlmProvider | None:
    processor_key = str(processor_name or "local").strip().lower()
    if processor_key in {"", "none", "local"}:
        return None
    if processor_key == "runpod":
        return resolve_async_llm_provider("runpod", model_name=model_name)
    raise ValueError(f"Unsupported processor: {processor_name}")


async def reason_implicit_signal_async(text: str, candidate_aspects: List[str], provider: AsyncLlmProvider, model_name: str, bypass_cache: bool = False) -> List[str]:
    prompt = f"""Task: Strictly Implicit Aspect Extraction
You are an expert linguistic analysis engine. Determine if the review segment IMPLIES any aspects from the allowed Canonical Ontology.

Review segment: "{text}"
Canonical Ontology: {candidate_aspects}

CRITICAL RULES:
1. PURELY IMPLICIT ONLY: If the text explicitly mentions the aspect by name or a direct synonym, you MUST NOT extract it. Only extract if the aspect is implied through context, effects, or evaluation without being named.
2. STRICT ONTOLOGY: If an implicit aspect is found, you MUST return exactly the aspect name from the Canonical Ontology. Do not modify the name.
3. OUTPUT FORMAT: Return ONLY the exact canonical aspect name. If no purely implicit aspect is present, return the exact word "none". No explanations, no quotes.

Extraction:"""
    
    cache_key = hashlib.md5(f"async:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key, bypass=bypass_cache)
    if cached_val:
        return [cached_val]

    try:
        paraphrase = await provider.generate(prompt, model_name, temperature=0.2, bypass_cache=bypass_cache)
        result = paraphrase.strip()
        GLOBAL_LLM_CACHE.set(cache_key, result)
        return [result]
    except Exception:
        return ["none"]


async def discover_novel_aspects_async(
    text: str, 
    excluded_aspects: List[str], 
    provider: AsyncLlmProvider, 
    model_name: str, 
    domain: str = "general",
    bypass_cache: bool = False
) -> List[dict[str, Any]]:
    """Phase 2: Open-Domain Discovery for aspects not in the registry."""
    prompt = f"""Task: Open-Domain Aspect Discovery
Domain: {domain}
Review Segment: "{text}"
Known Aspects to Ignore: {excluded_aspects}

Identify any NEW aspects mentioned or implied in the segment that are NOT in the ignore list.
For each new aspect:
1. Provide a concise Label (1-2 words).
2. Rate your Confidence (0.0 to 1.0).
3. Extract Evidence snippet from the text.

Return your response as a JSON list of objects:
[
  {{"label": "aspect name", "confidence": 0.85, "evidence": "text snippet"}}
]
If no new aspects are found, return [].
"""
    cache_key = hashlib.md5(f"discover:{model_name}:{prompt}".encode("utf-8")).hexdigest()
    cached_val = GLOBAL_LLM_CACHE.get(cache_key, bypass=bypass_cache)
    if cached_val:
        try: return json.loads(cached_val)
        except (json.JSONDecodeError, ValueError): pass

    try:
        res = await provider.generate(prompt, model_name, temperature=0.3, bypass_cache=bypass_cache)
        # Extract JSON from potential code blocks
        clean_res = res.strip()
        if "```json" in clean_res:
            clean_res = clean_res.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_res:
            clean_res = clean_res.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(clean_res)
        if not isinstance(parsed, list):
            parsed = []
        GLOBAL_LLM_CACHE.set(cache_key, json.dumps(parsed))
        return parsed
    except Exception:
        return []

