from __future__ import annotations

import hashlib
from typing import Dict, List


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def build_reviewlevel_record(source_file: str, idx: int, raw_text: str, clean_text: str, domain: str, rating, aspects: List[Dict], source_type: str = "raw") -> Dict:
    rid = f"rev_{stable_id(source_file, str(idx), clean_text)}"
    return {
        "review_id": rid,
        "source_file": source_file,
        "source_type": source_type,
        "domain": domain,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "rating": rating,
        "aspects": aspects,
        "is_augmented": source_type == "augmented",
        "split": "train",
    }
