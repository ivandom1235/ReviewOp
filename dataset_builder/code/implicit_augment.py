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
        for k in [
            "charge", "dropping", "drop", "wait", "late", "lags", "dismissive", "slow", "issue", "inconsistent",
            "overpriced", "worth", "deal", "fresh", "bland", "delicious", "cozy", "crowded", "helpful",
            "responsive", "intuitive", "confusing", "stable", "broke", "crashed",
        ]
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
    template_outputs: List[Dict] = []
    
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

    template_bank = {
        "performance": {
            "negative": "It felt noticeably sluggish once I started using it.",
            "positive": "It stayed quick and responsive in everyday use.",
        },
        "service": {
            "negative": "We spent too long waiting before anyone really helped us.",
            "positive": "Someone checked on us right when we needed it.",
        },
        "support_quality": {
            "negative": "Getting help felt more frustrating than it should have.",
            "positive": "Whenever help was needed, it was handled smoothly.",
        },
        "value": {
            "negative": "It did not feel worth the amount we paid.",
            "positive": "It felt like a solid value for the money.",
        },
        "product_quality": {
            "negative": "The flaws started showing sooner than expected.",
            "positive": "The overall quality came through immediately.",
        },
        "aesthetics": {
            "negative": "The look felt less polished in person.",
            "positive": "It looked even better up close.",
        },
        "experience": {
            "negative": "The overall atmosphere made it hard to settle in.",
            "positive": "The atmosphere made the whole experience enjoyable.",
        },
        "delivery_logistics": {
            "negative": "The wait dragged on much longer than promised.",
            "positive": "Everything arrived right when it was supposed to.",
        },
        "usability": {
            "negative": "Even basic tasks felt more confusing than they should have.",
            "positive": "It felt intuitive after just a minute or two.",
        },
        "reliability": {
            "negative": "Before long, it started acting up again.",
            "positive": "It kept working without any surprises.",
        },
        "taste": {
            "negative": "The flavor was flatter than I expected.",
            "positive": "The flavor stood out right away.",
        },
        "freshness": {
            "negative": "It tasted like it had been sitting around too long.",
            "positive": "Everything tasted freshly made.",
        },
        "portion_size": {
            "negative": "I was still hungry after finishing it.",
            "positive": "It ended up being more filling than expected.",
        },
        "service_speed": {
            "negative": "We were left waiting far longer than expected.",
            "positive": "Everything moved along without making us wait.",
        },
        "battery_life": {
            "negative": "I was already looking for a charger halfway through the day.",
            "positive": "It easily lasted through the day.",
        },
    }

    for aspect_info in base_record.get("aspects", []):
        aspect_key = str(aspect_info.get("aspect_canonical", "")).strip().lower()
        sentiment_key = "negative" if str(aspect_info.get("sentiment", "neutral")).lower() == "negative" else "positive"
        if aspect_key not in template_bank:
            if "_" in aspect_key:
                aspect_key = aspect_key.split("_", 1)[0]
        if aspect_key not in template_bank:
            continue
        txt = template_bank[aspect_key][sentiment_key]
        if txt.lower() in seen or not _quality_ok(txt, base_record, implicit_query_only=implicit_query_only):
            continue
        seen.add(txt.lower())
        aug = dict(base_record)
        aug["review_id"] = f"{base_record['review_id']}__tmpl_{len(template_outputs)+1}"
        aug["raw_text"] = txt
        aug["clean_text"] = txt
        aug["is_augmented"] = True
        aug["source_type"] = "augmented"
        aug["augmentation_type"] = "senticnet_style_template"
        aug["source_record_id"] = base_record["review_id"]
        aug["preserved_aspects"] = [aspect_info["aspect_canonical"]]
        aug["preserved_sentiments"] = [aspect_info["sentiment"]]
        template_outputs.append(aug)
        if len(template_outputs) >= 2:
            break

    outputs.extend(template_outputs)

    # 2. MANDATORY IMPLICIT BOOSTER (DEMO-GOLD PHASE)
    # We force at least one high-quality implicit rewrite if the record has a clear aspect.
    if not outputs and len(base_record.get("aspects", [])) > 0:
        asp = base_record["aspects"][0]["aspect_canonical"]
        sent = base_record["aspects"][0]["sentiment"]
        
        # Simple rule-based implicit templates for demo parity
        templates = {
            "performance": {
                "negative": "It was noticeably sluggish during use.",
                "positive": "Everything ran exactly as expected.",
            },
            "service": {
                "negative": "No one seemed to care we were there.",
                "positive": "We were taken care of the entire time.",
            },
            "support_quality": {
                "negative": "Getting help felt unnecessarily frustrating.",
                "positive": "Any time we needed help, it was handled smoothly.",
            },
            "value": {
                "negative": "I don't think it was worth the cost.",
                "positive": "It turned out to be a great deal.",
            },
            "product_quality": {
                "negative": "The overall quality felt disappointing right away.",
                "positive": "The overall quality came through immediately.",
            },
            "aesthetics": {
                "negative": "The appearance felt dated in person.",
                "positive": "It looks stunning in person.",
            },
            "experience": {
                "negative": "The whole atmosphere made it hard to relax.",
                "positive": "The atmosphere made the whole experience enjoyable.",
            },
            "delivery_logistics": {
                "negative": "The wait ended up being much longer than it should have been.",
                "positive": "Everything arrived right on time without any hassle.",
            },
            "usability": {
                "negative": "Even simple tasks felt more confusing than they should have.",
                "positive": "It felt intuitive from the first few minutes.",
            },
            "reliability": {
                "negative": "Before long, it started acting up again.",
                "positive": "It kept working without any surprises.",
            },
        }
        
        lookup_key = asp.lower()
        if "_" in lookup_key: lookup_key = lookup_key.split("_")[0]
        
        if lookup_key in templates:
            polarity = "negative" if sent == "negative" else "positive"
            txt = templates[lookup_key][polarity]
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

