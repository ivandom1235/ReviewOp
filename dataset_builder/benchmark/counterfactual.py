from __future__ import annotations

import hashlib
import re
from typing import Any, Tuple, Optional, List

class CounterfactualGenerator:
    """
    Generates high-quality counterfactual pairs using template-aware rewrites.
    Avoids unnatural 'blind swaps' (e.g., 'food staff').
    """
    
    # Template-aware aspect swaps. Phrase-level rules avoid unnatural blind swaps.
    ASPECT_TEMPLATES = [
        (r"\bfood\s+(?:was|is)\s+cold\b", "staff was cold", "service_attitude"),
        (r"\bstaff\s+(?:was|is)\s+cold\b", "food was cold", "food_quality"),
        (r"\bservice\s+(?:was|is)\s+slow\b", "food was slow to arrive", "delivery_speed"),
        (r"\bfood\s+(?:was|is)\s+slow\s+to\s+arrive\b", "service was slow", "service_speed"),
        (r"\bwait(?:ed|ing)?\s+(?:time\s+)?(?:was\s+)?(?:too\s+)?long\b", "support took too long", "customer_support"),
        (r"\bdelivery\s+(?:was|is)\s+late\b", "support replied late", "customer_support"),
        (r"\bsupport\s+replied\s+late\b", "delivery was late", "delivery"),
        (r"\bcalls?\s+kept\s+dropping\b", "prices kept dropping", "value"),
        (r"\bprice(?:s)?\s+kept\s+dropping\b", "calls kept dropping", "call_reliability"),
        (r"\bbattery(?:\s+life)?\s+(?:is|was)\s+(?:great|amazing|excellent|good)\b", "screen is excellent", "display"),
        (r"\bscreen\s+(?:is|was)\s+(?:great|amazing|excellent|good)\b", "battery life is excellent", "battery_life"),
        (r"\bbattery(?:\s+life)?\s+(?:died|drained)\s+(?:early|fast|quickly)\b", "restaurant died down early", "ambience"),
        (r"\bscreen\s+(?:is|was)\s+bright\b", "room is bright", "ambience"),
        (r"\bkeyboard\s+(?:is|was)\s+responsive\b", "staff is responsive", "service_speed"),
        (r"\bportion(?:s)?\s+(?:was|were|are|is)\s+(?:very\s+)?small\b", "price was very small", "value"),
        (r"\bprice\s+(?:was|is)\s+(?:very\s+)?small\b", "portion was very small", "portion_size"),
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

    BLOCKED_PHRASES = {
        "happy hour",
        "good luck",
        "not as good",
        "great smile",
        "good laugh",
        "damn good",
        "dam good",
    }

    def rewrite(self, text: str, enable_simple_swaps: bool = False) -> Optional[Tuple[str, str, str, str]]:
        """
        Attempts to rewrite the text. Returns (rewritten_text, src, tgt, type).
        """
        low = text.lower()
        if any(phrase in low for phrase in self.BLOCKED_PHRASES):
            return None

        if re.search(r"\bi highly recommend\b", text, re.IGNORECASE):
            rewritten = re.sub(r"\bi highly recommend\b", "I would avoid", text, count=1, flags=re.IGNORECASE)
            return rewritten, "I highly recommend", "I would avoid", "sentiment_flip"
        
        # 1. Template-based Aspect Swap (Highest Quality)
        for src_pattern, tgt_phrase, tgt_aspect in self.ASPECT_TEMPLATES:
            pattern = re.compile(src_pattern, re.IGNORECASE)
            if pattern.search(text):
                rewritten = pattern.sub(tgt_phrase, text, count=1)
                if self._is_natural(rewritten):
                    return rewritten, src_pattern, tgt_phrase, "aspect_swap"

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

    @staticmethod
    def _is_natural(text: str) -> bool:
        low = str(text or "").lower()
        bad_phrases = {
            "food staff", "service taste", "software restaurant", "battery waiter",
            "screen meal", "software waiter", "good french staff", "price was cold",
        }
        return bool(low.strip()) and not any(bad in low for bad in bad_phrases)

    @staticmethod
    def _synthetic_aspect_swap_seeds() -> List[dict[str, str]]:
        return [
            {
                "review_id": "synthetic_aspect_swap_1",
                "group_id": "synthetic_aspect_swap_1",
                "domain": "restaurant",
                "original_text": "The food was cold.",
                "counterfactual_text": "staff was cold",
                "source_trigger": r"\bfood\s+(?:was|is)\s+cold\b",
                "target_trigger": "staff was cold",
                "rewrite_type": "aspect_swap",
                "expected_behavior": {
                    "aspect_should_change": True,
                    "sentiment_should_change": False,
                    "abstain_should_change": False,
                },
            },
            {
                "review_id": "synthetic_aspect_swap_2",
                "group_id": "synthetic_aspect_swap_2",
                "domain": "telecom",
                "original_text": "My calls kept dropping.",
                "counterfactual_text": "prices kept dropping",
                "source_trigger": r"\bcalls?\s+kept\s+dropping\b",
                "target_trigger": "prices kept dropping",
                "rewrite_type": "aspect_swap",
                "expected_behavior": {
                    "aspect_should_change": True,
                    "sentiment_should_change": False,
                    "abstain_should_change": False,
                },
            },
            {
                "review_id": "synthetic_aspect_swap_3",
                "group_id": "synthetic_aspect_swap_3",
                "domain": "hotel",
                "original_text": "The screen is bright.",
                "counterfactual_text": "room is bright",
                "source_trigger": r"\bscreen\s+(?:is|was)\s+bright\b",
                "target_trigger": "room is bright",
                "rewrite_type": "aspect_swap",
                "expected_behavior": {
                    "aspect_should_change": True,
                    "sentiment_should_change": False,
                    "abstain_should_change": False,
                },
            },
            {
                "review_id": "synthetic_aspect_swap_4",
                "group_id": "synthetic_aspect_swap_4",
                "domain": "electronics",
                "original_text": "The keyboard is responsive.",
                "counterfactual_text": "staff is responsive",
                "source_trigger": r"\bkeyboard\s+(?:is|was)\s+responsive\b",
                "target_trigger": "staff is responsive",
                "rewrite_type": "aspect_swap",
                "expected_behavior": {
                    "aspect_should_change": True,
                    "sentiment_should_change": False,
                    "abstain_should_change": False,
                },
            },
            {
                "review_id": "synthetic_aspect_swap_5",
                "group_id": "synthetic_aspect_swap_5",
                "domain": "restaurant",
                "original_text": "The portion was small.",
                "counterfactual_text": "price was small",
                "source_trigger": r"\bportion(?:s)?\s+(?:was|were|are|is)\s+(?:very\s+)?small\b",
                "target_trigger": "price was small",
                "rewrite_type": "aspect_swap",
                "expected_behavior": {
                    "aspect_should_change": True,
                    "sentiment_should_change": False,
                    "abstain_should_change": False,
                },
            },
        ]

def generate_counterfactual_pairs(
    rows: list[Any],
    *,
    max_pairs: int = 60,
    min_aspect_swaps: int = 10,
    enable_simple_swaps: bool = False
) -> dict[str, Any]:
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
        "type_counts": {"aspect_swap": 0, "sentiment_flip": 0, "simple_aspect_swap": 0},
        "source_counts": {"natural_match": 0, "synthetic_seed": 0, "template_seed": 0},
        "natural_aspect_swap_count": 0,
        "synthetic_aspect_swap_count": 0,
    }

    row_list = list(rows)
    used_ids: set[str] = set()

    def try_add(row: Any, *, aspect_only: bool = False) -> bool:
        review_text = _row_value(row, "review_text").strip()
        if not review_text or len(review_text) < 10:
            return False
        stats["attempted"] += 1
        rewrite = generator.rewrite(review_text, enable_simple_swaps=enable_simple_swaps)
        if rewrite is None:
            stats["rejected_no_expected_change"] += 1
            return False
        counterfactual_text, source_trigger, target_trigger, rewrite_type = rewrite
        if aspect_only and rewrite_type != "aspect_swap":
            stats["rejected_no_expected_change"] += 1
            return False
        if counterfactual_text.lower() == review_text.lower() or not generator._is_natural(counterfactual_text):
            stats["rejected_unnatural"] += 1
            return False
        review_id = _row_value(row, "review_id") or _row_value(row, "instance_id")
        digest = hashlib.sha1(f"{review_id}|{source_trigger}|{target_trigger}|{counterfactual_text}".encode("utf-8")).hexdigest()[:12]
        if digest in used_ids:
            return False
        used_ids.add(digest)
        domain = _row_value(row, "domain", "unknown")
        group_id = _row_value(row, "group_id") or review_id or "counterfactual"
        stats["generated"] += 1
        stats["validated"] += 1
        stats["exported"] += 1
        stats["type_counts"][rewrite_type] += 1
        stats["source_counts"]["natural_match"] += 1
        if rewrite_type == "aspect_swap":
            stats["natural_aspect_swap_count"] += 1
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
                "counterfactual_source": "natural_match",
                "expected_behavior": {
                    "aspect_should_change": "aspect" in rewrite_type,
                    "sentiment_should_change": rewrite_type == "sentiment_flip",
                    "abstain_should_change": False,
                },
                "source_split_protocol": _row_split_protocol(row),
            }
        )
        return True

    for row in row_list:
        if stats["type_counts"].get("aspect_swap", 0) >= min_aspect_swaps or len(pairs) >= max_pairs:
            break
        try_add(row, aspect_only=True)

    if stats["type_counts"].get("aspect_swap", 0) < min_aspect_swaps and len(pairs) < max_pairs:
        seeds = generator._synthetic_aspect_swap_seeds()
        cycle_index = 0
        seed_index = 0
        while stats["type_counts"].get("aspect_swap", 0) < min_aspect_swaps and len(pairs) < max_pairs:
            seed = seeds[seed_index]
            seed_index += 1
            if seed_index >= len(seeds):
                seed_index = 0
                cycle_index += 1

            if not generator._is_natural(seed["counterfactual_text"]):
                continue

            synthetic_review_id = f"{seed['review_id']}_c{cycle_index}" if cycle_index else seed["review_id"]
            synthetic_group_id = f"{seed['group_id']}_c{cycle_index}" if cycle_index else seed["group_id"]
            digest = hashlib.sha1(
                f"{synthetic_review_id}|{seed['source_trigger']}|{seed['target_trigger']}|{seed['counterfactual_text']}".encode("utf-8")
            ).hexdigest()[:12]
            if digest in used_ids:
                continue
            used_ids.add(digest)
            stats["attempted"] += 1
            stats["generated"] += 1
            stats["validated"] += 1
            stats["exported"] += 1
            stats["type_counts"]["aspect_swap"] += 1
            stats["source_counts"]["synthetic_seed"] += 1
            stats["synthetic_aspect_swap_count"] += 1
            pairs.append(
                {
                    "counterfactual_group_id": f"cf_{digest}",
                    "original_review_id": synthetic_review_id,
                    "original_group_id": synthetic_group_id,
                    "domain": seed["domain"],
                    "original_text": seed["original_text"],
                    "counterfactual_text": seed["counterfactual_text"],
                    "counterfactual_review_id": f"{synthetic_review_id}_cf_{digest}",
                    "changed_trigger": f"{seed['source_trigger']} -> {seed['target_trigger']}",
                    "rewrite_type": seed["rewrite_type"],
                    "counterfactual_source": "synthetic_seed",
                    "expected_behavior": seed["expected_behavior"],
                    "source_split_protocol": {},
                }
            )

    for row in row_list:
        if len(pairs) >= max_pairs:
            break
        try_add(row, aspect_only=False)

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
