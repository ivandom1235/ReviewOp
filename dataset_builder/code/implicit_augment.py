from __future__ import annotations

import re
from typing import Dict, List

from llm_utils import LLMClient
from prompts import IMPLICIT_REWRITE_PROMPT


def _heuristic_rewrite(text: str) -> List[Dict]:
    patterns = [
        ("battery life", "I had to charge it twice in one day", "explicit_to_implicit"),
        ("delivery", "it showed up much later than promised", "explicit_to_implicit"),
        ("service", "we waited nearly half an hour before anyone came", "explicit_to_implicit"),
        ("staff", "the interaction felt dismissive", "explicit_to_implicit"),
        ("network", "calls kept dropping every few minutes", "explicit_to_implicit"),
    ]
    out = []
    t = text
    for src, dst, typ in patterns:
        if src in t.lower():
            out.append({"text": re.sub(re.escape(src), dst, t, flags=re.IGNORECASE), "augmentation_type": typ})
            break
    out.append({"text": text + " In day-to-day use, that started to show.", "augmentation_type": "mixed_implicit"})
    out.append({"text": text + " Over time, the issues became hard to ignore.", "augmentation_type": "symptom_summary"})
    return out


def _aspect_heads(aspect: str) -> List[str]:
    parts = aspect.replace("other_", "").split("_")
    return [p for p in parts if p and p not in {"other", "general"}]


def _quality_ok(text: str, base_record: Dict, implicit_query_only: bool = True) -> bool:
    if len(text.split()) < 8:
        return False
    low = text.lower()
    aspects = [a.get("aspect_canonical", "") for a in base_record.get("aspects", [])]
    sentiments = [a.get("sentiment", "neutral") for a in base_record.get("aspects", [])]

    if not aspects:
        return False

    symptom_ok = any(
        k in low
        for k in ["charge", "dropping", "drop", "wait", "late", "lags", "dismissive", "slow", "issue", "inconsistent"]
    )
    if not symptom_ok:
        return False

    if implicit_query_only:
        # no direct explicit aspect head leakage
        leakage_hits = 0
        for a in aspects:
            for h in _aspect_heads(a):
                if len(h) >= 4 and h in low:
                    leakage_hits += 1
        if leakage_hits > 2:
            return False

    # preserve sentiment direction cues lightly
    if any(s == "negative" for s in sentiments) and not any(k in low for k in ["late", "slow", "drop", "barely", "dismissive", "bad", "issue"]):
        return False
    return True


def generate_augmented_texts(text: str, llm: LLMClient | None = None) -> List[Dict]:
    if llm is None:
        return _heuristic_rewrite(text)
    payload = llm.json_completion(IMPLICIT_REWRITE_PROMPT + f"\nReview: {text}")
    rows = payload.get("rewrites", []) if isinstance(payload, dict) else []
    valid = []
    for r in rows:
        txt = str(r.get("text", "")).strip()
        typ = str(r.get("augmentation_type", "implicit_rewrite")).strip()
        if txt:
            valid.append({"text": txt, "augmentation_type": typ})
    return valid or _heuristic_rewrite(text)


def build_augmented_records(base_record: Dict, llm: LLMClient | None = None, implicit_query_only: bool = True) -> List[Dict]:
    outputs = []
    seen = set()
    for item in generate_augmented_texts(base_record["clean_text"], llm=llm):
        txt = item["text"].strip()
        if txt.lower() in seen:
            continue
        seen.add(txt.lower())
        if not _quality_ok(txt, base_record, implicit_query_only=implicit_query_only):
            continue

        aug = dict(base_record)
        aug["review_id"] = f"{base_record['review_id']}__aug_{len(outputs)+1}"
        aug["raw_text"] = txt
        aug["clean_text"] = txt
        aug["is_augmented"] = True
        aug["source_type"] = "augmented"
        aug["augmentation_type"] = item["augmentation_type"]
        aug["source_record_id"] = base_record["review_id"]
        aug["preserved_aspects"] = [a["aspect_canonical"] for a in base_record.get("aspects", [])]
        aug["preserved_sentiments"] = [a["sentiment"] for a in base_record.get("aspects", [])]
        outputs.append(aug)

    if not outputs:
        txt = base_record["clean_text"].strip() + " In regular use, the issues were still noticeable."
        if _quality_ok(txt, base_record, implicit_query_only=False):
            aug = dict(base_record)
            aug["review_id"] = f"{base_record['review_id']}__aug_1"
            aug["raw_text"] = txt
            aug["clean_text"] = txt
            aug["is_augmented"] = True
            aug["source_type"] = "augmented"
            aug["augmentation_type"] = "fallback_implicit"
            aug["source_record_id"] = base_record["review_id"]
            aug["preserved_aspects"] = [a["aspect_canonical"] for a in base_record.get("aspects", [])]
            aug["preserved_sentiments"] = [a["sentiment"] for a in base_record.get("aspects", [])]
            outputs.append(aug)

    return outputs
