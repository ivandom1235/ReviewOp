from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_count(text: str) -> int:
    return len([t for t in re.split(r"\W+", text.lower()) if t])


def jaccard(a: str, b: str) -> float:
    aa = set([t for t in re.split(r"\W+", a.lower()) if t])
    bb = set([t for t in re.split(r"\W+", b.lower()) if t])
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def standardize_rating(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        v = float(str(value).strip())
    except Exception:
        return None
    if v <= 0:
        return None
    if v <= 5:
        return int(round(v))
    if v <= 10:
        return int(round(v / 2))
    return 5


def clean_records(
    records: List[Dict],
    min_review_length: int,
    near_dup_threshold: float,
    dedupe_exact: bool = True,
) -> Tuple[List[Dict], Dict[str, int]]:
    cleaned: List[Dict] = []
    removed = {"empty": 0, "too_short": 0, "exact_dup": 0, "near_dup": 0}
    seen_hashes = set()
    prior_tokens: List[set[str]] = []
    enable_near_dup = near_dup_threshold < 1.0

    for r in records:
        raw = normalize_text(r.get("raw_text", ""))
        text = normalize_text(r.get("clean_text", raw))
        if not text:
            removed["empty"] += 1
            continue
        if token_count(text) < min_review_length:
            removed["too_short"] += 1
            continue
        digest = hashlib.sha1(text.lower().encode("utf-8")).hexdigest()
        if dedupe_exact and digest in seen_hashes:
            removed["exact_dup"] += 1
            continue
        if enable_near_dup:
            cur_tokens = set([t for t in re.split(r"\W+", text.lower()) if t])
            near = False
            for prev_tokens in prior_tokens[-500:]:
                if not prev_tokens or not cur_tokens:
                    continue
                score = len(cur_tokens & prev_tokens) / len(cur_tokens | prev_tokens)
                if score >= near_dup_threshold:
                    near = True
                    break
            if near:
                removed["near_dup"] += 1
                continue
            prior_tokens.append(cur_tokens)
        if dedupe_exact:
            seen_hashes.add(digest)
        r["clean_text"] = text
        cleaned.append(r)

    return cleaned, removed
