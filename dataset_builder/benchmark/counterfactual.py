from __future__ import annotations

import hashlib
import re
from typing import Any, Tuple, Optional, List

class CounterfactualGenerator:
    """
    Generates high-quality counterfactual pairs using template-aware rewrites.
    Avoids unnatural 'blind swaps' (e.g., 'food staff').
    """
    
    # Template-aware aspect swaps
    ASPECT_TEMPLATES = [
        # Original Trigger -> Counterfactual Trigger -> Category
        ("food was cold", "staff was cold", "service_attitude"),
        ("portions are small", "prices are small", "value"),
        ("calls kept dropping", "prices kept dropping", "value"),
        ("battery life is amazing", "screen is amazing", "display"),
        ("delivery was late", "support replied late", "customer_support"),
        ("battery died early", "restaurant died down early", "ambience"),
        ("screen is bright", "room is bright", "ambience"),
        ("keyboard is responsive", "staff is responsive", "service_speed"),
    ]

    # Sentiment flips (more robust)
    SENTIMENT_FLIPS = [
        (r"\bgood\b", "bad"),
        (r"\bgreat\b", "terrible"),
        (r"\bfast\b", "slow"),
        (r"\bexcellent\b", "poor"),
        (r"\bhappy\b", "unhappy"),
        (r"\brecommend\b", "avoid"),
        (r"\bloved\b", "hated"),
        (r"\bperfect\b", "flawed"),
    ]

    def rewrite(self, text: str, enable_simple_swaps: bool = False) -> Optional[Tuple[str, str, str, str]]:
        """
        Attempts to rewrite the text. Returns (rewritten_text, src, tgt, type).
        """
        low = text.lower()
        
        # 1. Template-based Aspect Swap (Highest Quality)
        for src_phrase, tgt_phrase, tgt_aspect in self.ASPECT_TEMPLATES:
            if src_phrase in low:
                # Use regex for case-insensitive replacement of the exact phrase
                pattern = re.compile(re.escape(src_phrase), re.IGNORECASE)
                rewritten = pattern.sub(tgt_phrase, text)
                return rewritten, src_phrase, tgt_phrase, "aspect_swap"

        # 2. Robust Sentiment Flip
        for src_pattern, tgt_word in self.SENTIMENT_FLIPS:
            pattern = re.compile(src_pattern, re.IGNORECASE)
            if pattern.search(text):
                rewritten = pattern.sub(tgt_word, text, count=1)
                return rewritten, src_pattern.replace(r"\b", ""), tgt_word, "sentiment_flip"

        # 3. Simple Aspect Swap (with safety gate)
        if enable_simple_swaps:
            simple_swaps = [
                ("battery", "screen"), ("service", "food"), ("price", "quality"),
                ("delivery", "support"), ("calls", "prices"), ("food", "staff")
            ]
            for src, tgt in simple_swaps:
                pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
                if pattern.search(text):
                    rewritten = pattern.sub(tgt, text, count=1)
                    # Hard Quality Gate: Reject unnatural word pairings
                    bad_phrases = ["food staff", "service taste", "battery waiter", "screen meal", "software waiter", "good french staff"]
                    if any(bad in rewritten.lower() for bad in bad_phrases):
                        continue
                    return rewritten, src, tgt, "simple_aspect_swap"

        return None

def generate_counterfactual_pairs(rows: list[Any], *, max_pairs: int = 60, enable_simple_swaps: bool = False) -> dict[str, Any]:
    generator = CounterfactualGenerator()
    pairs: list[dict[str, Any]] = []
    
    # Tracking for metrics
    stats = {
        "attempted": 0,
        "generated": 0,
        "exported": 0,
        "validated": 0,
        "rejected_no_expected_change": 0,
        "rejected_unnatural": 0,
        "type_counts": {"aspect_swap": 0, "sentiment_flip": 0, "simple_aspect_swap": 0}
    }

    for row in rows:
        review_text = _row_value(row, "review_text").strip()
        if not review_text or len(review_text) < 10:
            continue
            
        stats["attempted"] += 1
        rewrite = generator.rewrite(review_text, enable_simple_swaps=enable_simple_swaps)
        
        if rewrite is None:
            stats["rejected_no_expected_change"] += 1
            continue
            
        counterfactual_text, source_trigger, target_trigger, rewrite_type = rewrite
        
        # Additional safety check
        if counterfactual_text.lower() == review_text.lower():
            stats["rejected_no_expected_change"] += 1
            continue
            
        stats["generated"] += 1
        stats["validated"] += 1
        stats["exported"] += 1
        stats["type_counts"][rewrite_type] += 1

        review_id = _row_value(row, "review_id") or _row_value(row, "instance_id")
        domain = _row_value(row, "domain", "unknown")
        group_id = _row_value(row, "group_id") or review_id or "counterfactual"
        
        digest = hashlib.sha1(f"{review_id}|{source_trigger}|{target_trigger}|{counterfactual_text}".encode("utf-8")).hexdigest()[:12]
        
        pairs.append(
            {
                "counterfactual_group_id": f"cf_{digest}",
                "original_review_id": review_id,
                "original_group_id": group_id,
                "domain": domain,
                "original_text": review_text,
                "counterfactual_text": counterfactual_text,
                "counterfactual_review_id": f"{review_id}_cf_{digest}",
                "changed_trigger": f"{source_trigger} -> {target_trigger}",
                "rewrite_type": rewrite_type,
                "expected_behavior": {
                    "aspect_should_change": "aspect" in rewrite_type,
                    "sentiment_should_change": rewrite_type == "sentiment_flip",
                    "abstain_should_change": False,
                },
                "source_split_protocol": _row_split_protocol(row),
            }
        )
        
        if len(pairs) >= max_pairs:
            break
            
    return {
        "pairs": pairs,
        "stats": stats
    }

def _row_value(row: Any, key: str, default: str = "") -> str:
    if isinstance(row, dict):
        return str(row.get(key, default) or default)
    return str(getattr(row, key, default) or default)

def _row_split_protocol(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        payload = row.get("split_protocol")
    else:
        payload = getattr(row, "split_protocol", None)
    return dict(payload) if isinstance(payload, dict) else {}
