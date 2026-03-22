from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from config import BuilderConfig, llm_available


class LLMClient:
    def __init__(self, cfg: BuilderConfig):
        self.cfg = cfg

    def _post_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=self.cfg.llm.timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)

    def _request_openai(self, prompt: str) -> str:
        url = self.cfg.llm.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.llm.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.llm.openai_model,
            "messages": [
                {"role": "system", "content": "Return concise, valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        out = self._post_json(url, payload, headers)
        return out["choices"][0]["message"]["content"]

    def _request_groq(self, prompt: str) -> str:
        url = self.cfg.llm.groq_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.llm.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.llm.groq_model,
            "messages": [
                {"role": "system", "content": "Return concise, valid JSON when asked."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        out = self._post_json(url, payload, headers)
        return out["choices"][0]["message"]["content"]

    def _request_anthropic(self, prompt: str) -> str:
        url = self.cfg.llm.anthropic_base_url.rstrip("/") + "/v1/messages"
        headers = {
            "x-api-key": self.cfg.llm.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.cfg.llm.anthropic_model,
            "max_tokens": 800,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}],
        }
        out = self._post_json(url, payload, headers)
        content = out.get("content", [])
        text = "".join(block.get("text", "") for block in content if isinstance(block, dict))
        return text

    def completion(self, prompt: str) -> Optional[str]:
        if not llm_available(self.cfg):
            return None
        last_error: Exception | None = None
        for attempt in range(1, self.cfg.llm.max_retries + 1):
            try:
                if self.cfg.llm.provider == "anthropic":
                    return self._request_anthropic(prompt)
                if self.cfg.llm.provider == "groq":
                    return self._request_groq(prompt)
                return self._request_openai(prompt)
            except Exception as exc:
                last_error = exc
                if attempt == self.cfg.llm.max_retries:
                    print(
                        f"[LLM warning] provider={self.cfg.llm.provider} failed after {self.cfg.llm.max_retries} attempts: {type(last_error).__name__}: {last_error}"
                    )
                    return None
                time.sleep(1.2 * attempt)
        return None

    def json_completion(self, prompt: str) -> Dict[str, Any]:
        text = self.completion(prompt)
        if not text:
            return {}
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                    return data if isinstance(data, dict) else {}
                except Exception:
                    return {}
            return {}
