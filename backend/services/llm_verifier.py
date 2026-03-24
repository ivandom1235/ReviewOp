# proto/backend/services/llm_verifier.py
from __future__ import annotations

import json
from typing import Any, Dict, List

import requests

from core.config import settings


PredictionLike = Dict[str, Any]


class LLMVerifier:
    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model_name
        self.timeout = settings.llm_timeout_seconds

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

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        predictions = parsed.get("predictions", [])

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