"""
RunPod Flash inference client for ReviewOp.

Hybrid system: uses RunPod Flash GPU for sentiment classification by default,
falls back to local Seq2SeqEngine (flan-t5-base on CPU) when Flash is unavailable
or times out.

Architecture:
  1. On backend startup, warmup() sends a lightweight ping to wake the endpoint.
  2. Inference requests try Flash first (with configurable timeout).
  3. If Flash fails/times out, falls back to local Seq2SeqEngine.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded Flash components (only import if flash_enabled)
_flash_endpoint = None
_flash_initialized = False


def _get_flash_endpoint():
    """Lazily initialize the Flash endpoint definition."""
    global _flash_endpoint, _flash_initialized
    if _flash_initialized:
        return _flash_endpoint
    _flash_initialized = True

    if not settings.flash_enabled:
        logger.info("RunPod Flash is disabled (FLASH_ENABLED=false)")
        return None

    try:
        from runpod_flash import Endpoint, GpuType
        gpu_map = {
            "NVIDIA_GEFORCE_RTX_4090": GpuType.NVIDIA_GEFORCE_RTX_4090,
            "ADA_24": GpuType.ADA_24 if hasattr(GpuType, "ADA_24") else GpuType.NVIDIA_GEFORCE_RTX_4090,
            "AMPERE_80": GpuType.AMPERE_80 if hasattr(GpuType, "AMPERE_80") else GpuType.NVIDIA_GEFORCE_RTX_4090,
        }
        gpu_type = gpu_map.get(settings.flash_gpu_type, GpuType.NVIDIA_GEFORCE_RTX_4090)

        _flash_endpoint = Endpoint(
            name="reviewop-sentiment",
            gpu=gpu_type,
            workers=(0, settings.flash_max_workers),
            idle_timeout=settings.flash_idle_timeout,
            flashboot=True,
            dependencies=["torch", "transformers"],
        )
        logger.info(
            "RunPod Flash endpoint configured: gpu=%s, workers=(0,%d), idle_timeout=%ds",
            settings.flash_gpu_type,
            settings.flash_max_workers,
            settings.flash_idle_timeout,
        )
        return _flash_endpoint
    except ImportError:
        logger.warning("runpod-flash not installed. Flash inference disabled.")
        return None
    except Exception as exc:
        logger.warning("Failed to initialize Flash endpoint: %s", exc)
        return None


# The decorated function for remote execution
async def _remote_classify_sentiment(data: dict[str, Any]) -> dict[str, Any]:
    """
    This function body runs on the RunPod GPU worker.
    All imports MUST be inside the function (cloudpickle requirement).
    """
    ep = _get_flash_endpoint()
    if ep is None:
        raise RuntimeError("Flash endpoint not available")

    result = await ep.run(data)
    await result.wait(timeout=settings.flash_timeout_seconds)
    if result.error:
        raise RuntimeError(f"Flash job error: {result.error}")
    return result.output


class FlashInferenceClient:
    """Hybrid RunPod Flash client with local CPU fallback."""

    def __init__(self) -> None:
        self._endpoint_ready = False
        self._local_engine = None
        self._local_engine_loaded = False

    def _get_local_engine(self):
        """Lazy-load the local Seq2SeqEngine for fallback."""
        if not self._local_engine_loaded:
            self._local_engine_loaded = True
            try:
                from services.seq2seq_infer import Seq2SeqEngine
                self._local_engine = Seq2SeqEngine.load()
                logger.info("Local fallback Seq2SeqEngine loaded successfully")
            except Exception as exc:
                logger.error("Failed to load local fallback engine: %s", exc)
                self._local_engine = None
        return self._local_engine

    async def warmup(self) -> None:
        """
        Called on backend startup. Sends a health ping to wake the endpoint.

        Strategy: By pre-warming on startup, the GPU worker is provisioned
        during initialization (~30-60s cold start). By the time a user
        submits a prompt, the endpoint should be ready (~2-3s response).
        """
        ep = _get_flash_endpoint()
        if ep is None:
            logger.info("Flash warmup skipped (endpoint not configured)")
            self._endpoint_ready = False
            return

        try:
            warmup_data = {
                "evidence_text": "The product works great",
                "aspect": "quality",
            }
            result = await asyncio.wait_for(
                _remote_classify_sentiment(warmup_data),
                timeout=settings.flash_timeout_seconds,
            )
            self._endpoint_ready = True
            logger.info("RunPod Flash endpoint warmed up: %s", result)
        except asyncio.TimeoutError:
            logger.warning(
                "RunPod Flash warmup timed out after %ds — will retry on first request",
                settings.flash_timeout_seconds,
            )
            self._endpoint_ready = False
        except Exception as exc:
            logger.warning("RunPod Flash warmup failed: %s — using local fallback", exc)
            self._endpoint_ready = False

    async def classify_sentiment(
        self,
        evidence_text: str,
        aspect: str,
    ) -> tuple[str, float]:
        """
        Classify sentiment via Flash GPU, with automatic local fallback.

        Returns:
            (label, confidence) where label is one of positive/neutral/negative.
        """
        ep = _get_flash_endpoint()
        if ep is not None and settings.flash_enabled:
            try:
                data = {"evidence_text": evidence_text, "aspect": aspect}
                result = await asyncio.wait_for(
                    _remote_classify_sentiment(data),
                    timeout=settings.flash_timeout_seconds,
                )
                label = str(result.get("sentiment", "neutral")).strip().lower()
                confidence = float(result.get("confidence", 0.85))
                valid = {"positive", "neutral", "negative"}
                if label in valid:
                    self._endpoint_ready = True
                    return label, confidence
                return "neutral", 0.5
            except asyncio.TimeoutError:
                logger.warning("Flash timeout (%ds), falling back to local", settings.flash_timeout_seconds)
            except Exception as exc:
                logger.warning("Flash error: %s — falling back to local", exc)

        return self._local_fallback(evidence_text, aspect)

    def _local_fallback(self, evidence_text: str, aspect: str) -> tuple[str, float]:
        """Use the local Seq2SeqEngine (flan-t5-base CPU) as fallback."""
        engine = self._get_local_engine()
        if engine is None:
            logger.error("No inference engine available (Flash down + local failed)")
            return "neutral", 0.0
        return engine.classify_sentiment_with_confidence(evidence_text, aspect)

    @property
    def is_flash_ready(self) -> bool:
        return self._endpoint_ready and settings.flash_enabled

    def status(self) -> dict[str, Any]:
        return {
            "flash_enabled": settings.flash_enabled,
            "flash_ready": self._endpoint_ready,
            "flash_gpu_type": settings.flash_gpu_type,
            "flash_max_workers": settings.flash_max_workers,
            "flash_timeout_seconds": settings.flash_timeout_seconds,
            "local_fallback_loaded": self._local_engine_loaded,
        }
