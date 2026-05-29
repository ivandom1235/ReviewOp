from __future__ import annotations
import os
import time
import threading
from collections import deque
import google.auth
from google import genai
from .base_client import BaseLLMClient
from ..config import BuilderConfig

class RateLimiter:
    """Thread-safe sliding window rate limiter."""
    def __init__(self, rpm_limit: int):
        self.rpm_limit = rpm_limit
        self.requests = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        if self.rpm_limit <= 0:
            return
            
        while True:
            with self.lock:
                now = time.time()
                # Remove requests older than 60 seconds
                while self.requests and self.requests[0] < now - 60:
                    self.requests.popleft()
                
                if len(self.requests) < self.rpm_limit:
                    self.requests.append(now)
                    return
                
                # Calculate how long to wait until the oldest request expires
                sleep_time = self.requests[0] + 60 - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)

class GeminiClient(BaseLLMClient):
    # Class-level rate limiter to share across all instances if needed, 
    # but usually one config = one client instance in dataset_builder.
    _limiter = RateLimiter(rpm_limit=100)

    def __init__(self, cfg: BuilderConfig):
        super().__init__(cfg)
        _, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "No Google Cloud project found. Configure ADC or set GOOGLE_CLOUD_PROJECT."
            )
        # Use Vertex AI as verified in tests
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    def _generate_inner(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        # Enforce the 100 RPM limit
        self._limiter.wait_if_needed()
        
        config_args = {}
        if system_prompt:
            config_args["system_instruction"] = system_prompt
        
        # Merge kwargs into config_args if they are generation parameters
        if "temperature" in kwargs:
            config_args["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            config_args["max_output_tokens"] = kwargs.pop("max_tokens")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config_args,
            **kwargs
        )
        return response.text or ""
