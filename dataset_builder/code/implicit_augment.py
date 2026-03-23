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
    
    # 1. Main Augmentation Loop
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

    # 2. MANDATORY IMPLICIT BOOSTER (DEMO-GOLD PHASE)
    # We force at least one high-quality implicit rewrite if the record has a clear aspect.
    if not outputs and len(base_record.get("aspects", [])) > 0:
        asp = base_record["aspects"][0]["aspect_canonical"]
        sent = base_record["aspects"][0]["sentiment"]
        
        # Simple rule-based implicit templates for demo parity
        templates = {
            "performance": ["It was noticeably sluggish during use.", "Everything ran exactly as expected."],
            "service": ["We were taken care of the entire time.", "No one seemed to care we were there."],
            "value": ["It turned out to be a great deal.", "I don't think it was worth the cost."],
            "product_quality": ["The build feels very cheap.", "It is clearly made of premium materials."],
            "aesthetics": ["It looks stunning in person.", "The appearance is quite dated."]
        }
        
        lookup_key = asp.lower()
        if "_" in lookup_key: lookup_key = lookup_key.split("_")[0]
        
        if lookup_key in templates:
            txt = templates[lookup_key][0] if sent == "negative" else templates[lookup_key][1]
            aug = dict(base_record)
            aug["review_id"] = f"{base_record['review_id']}__boot_1"
            aug["raw_text"] = txt
            aug["clean_text"] = txt
            aug["is_augmented"] = True
            aug["source_type"] = "augmented"
            aug["augmentation_type"] = "demo_booster_implicit"
            aug["source_record_id"] = base_record["review_id"]
            aug["preserved_aspects"] = [asp]
            aug["preserved_sentiments"] = [sent]
            outputs.append(aug)

    return outputs


def diversify_paraphrases(text: str, aspect: str, sentiment: str, llm: LLMClient) -> List[str]:
    prompt = f"""Given the review sentence: "{text}"
It expresses a {sentiment} sentiment about the implicit aspect "{aspect}".
Generate 3 distinct paraphrases that convey the exact same sentiment about the same aspect, but using different surface vocabulary. Do not explicitly name the aspect.
Return as JSON strictly: {{"paraphrases": ["p1", "p2", "p3"]}}"""
    try:
        resp = llm.json_completion(prompt)
        if isinstance(resp, dict):
            return resp.get("paraphrases", [])
    except Exception:
        pass
    return []

def cross_domain_transfer(text: str, source_domain: str, target_domain: str, llm: LLMClient) -> str:
    prompt = f"""Rewrite this review sentence from the {source_domain} domain into the {target_domain} domain's register.
Preserve the underlying sentiment and the abstract aspect being discussed.
Original: "{text}"
Return as JSON strictly: {{"transferred_text": "..."}}"""
    try:
        resp = llm.json_completion(prompt)
        if isinstance(resp, dict):
            return resp.get("transferred_text", text)
    except Exception:
        pass
    return text

