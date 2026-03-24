# proto/backend/services/llm_verifier.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from core.config import settings


PredictionLike = Dict[str, Any]

logger = logging.getLogger(__name__)


class LLMVerifier:
    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=settings.llm_timeout_seconds,
        )
        self.model = settings.llm_model_name

    def verify(
        self,
        review_text: str,
        explicit_predictions: List[PredictionLike],
        implicit_predictions: List[PredictionLike],
        merged_predictions: List[PredictionLike],
    ) -> List[PredictionLike]:
        if not settings.enable_llm_verifier:
            return merged_predictions

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded review aspect verifier.\n"
                    "Reason internally step by step, but DO NOT reveal chain-of-thought.\n"
                    "Return JSON only.\n"
                    "Task:\n"
                    "1. Keep only aspect predictions supported by the review text.\n"
                    "2. Remove weak or duplicate predictions.\n"
                    "3. For implicit aspects, keep them only if clearly inferable from the wording.\n"
                    "4. You may merge wording variants.\n"
                    "5. Do not invent unsupported aspects.\n"
                    "6. Each final item must include a short public rationale, not hidden reasoning.\n"
                    "Allowed sentiments: positive, neutral, negative.\n"
                    "Output schema:\n"
                    "{\n"
                    '  "predictions": [\n'
                    "    {\n"
                    '      "aspect_raw": str,\n'
                    '      "aspect_cluster": str,\n'
                    '      "sentiment": "positive|neutral|negative",\n'
                    '      "confidence": float,\n'
                    '      "source": "explicit|implicit|verified",\n'
                    '      "rationale": str,\n'
                    '      "evidence_spans": [{"start_char": int, "end_char": int, "snippet": str}]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "review_text": review_text,
                        "explicit_predictions": explicit_predictions,
                        "implicit_predictions": implicit_predictions,
                        "merged_predictions": merged_predictions,
                        "max_predictions": settings.max_verified_predictions,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            predictions = parsed.get("predictions", [])
        except Exception as exc:
            logger.warning(
                "LLM verifier unavailable; returning merged predictions unchanged: %s",
                exc,
            )
            return merged_predictions

        final_rows: List[PredictionLike] = []
        for row in predictions:
            final_rows.append(
                {
                    "aspect_raw": str(row.get("aspect_raw") or "").strip(),
                    "aspect_cluster": str(
                        row.get("aspect_cluster") or row.get("aspect_raw") or ""
                    ).strip(),
                    "sentiment": str(row.get("sentiment") or "neutral").strip().lower(),
                    "confidence": float(row.get("confidence", 0.0)),
                    "source": str(row.get("source") or "verified").strip().lower(),
                    "rationale": str(row.get("rationale") or "").strip(),
                    "evidence_spans": row.get("evidence_spans") or [],
                }
            )

        # Fallback if model returns nothing
        return [x for x in final_rows if x["aspect_raw"]] or merged_predictions
