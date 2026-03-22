from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

DOMAIN_KEYWORDS = {
    "electronics": ["battery", "screen", "charger", "laptop", "phone", "camera"],
    "restaurant": ["food", "taste", "waiter", "service", "menu", "dish"],
    "hotel": ["room", "checkin", "reception", "bed", "stay"],
    "delivery": ["delivery", "courier", "package", "shipping"],
    "healthcare": ["doctor", "clinic", "hospital", "appointment"],
    "telecom": ["network", "signal", "call", "data", "plan"],
    "ecommerce": ["seller", "return", "refund", "cart", "order"],
    "services": ["support", "agent", "staff", "response", "booking"],
}


def infer_domain(source_name: str, metadata_domain: str, text: str) -> str:
    source_l = source_name.lower()
    metadata_l = metadata_domain.strip().lower() if metadata_domain else ""
    text_l = text.lower()
    score = Counter()
    for d, kws in DOMAIN_KEYWORDS.items():
        score[d] += sum(1 for k in kws if re.search(rf"\b{re.escape(k)}\b", text_l))
        if d in source_l:
            score[d] += 2
        if metadata_l == d:
            score[d] += 1
        elif metadata_l and metadata_l in kws:
            score[d] += 1
    best, cnt = (score.most_common(1)[0] if score else ("general", 0))
    return best if cnt > 0 else "general"


def domain_stats(records: List[Dict]) -> Dict[str, int]:
    out = Counter(r.get("domain", "general") for r in records)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))
