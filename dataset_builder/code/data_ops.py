from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, List


def _stable_group_key(record: Dict) -> str:
    text = str(record.get("clean_text", "")).strip().lower()
    if text:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:14]
    base = record.get("group_id") or record.get("review_id") or ""
    return hashlib.sha1(str(base).encode("utf-8")).hexdigest()[:14]


def assign_splits(records: List[Dict], preserve_official_splits: bool, ratios: Dict[str, float], seed: int = 42) -> List[Dict]:
    if preserve_official_splits and any(str(r.get("official_split", "")).lower() in {"train", "val", "test"} for r in records):
        for r in records:
            sp = str(r.get("official_split", "")).lower()
            r["split"] = sp if sp in {"train", "val", "test"} else "train"
        return records

    grouped: Dict[str, List[Dict]] = {}
    for r in records:
        grouped.setdefault(_stable_group_key(r), []).append(r)

    def _implicit_score(key: str) -> float:
        rows = grouped.get(key, [])
        aspects = [a for r in rows for a in r.get("aspects", [])]
        if not aspects:
            return 0.0
        return sum(1 for a in aspects if a.get("aspect_type") == "implicit") / max(1, len(aspects))

    keys = sorted(grouped.keys(), key=lambda k: (-_implicit_score(k), k))
    rng = random.Random(seed)
    rng.shuffle(keys)
    # Reorder by implicit density buckets while keeping reproducible randomization within each bucket.
    buckets: Dict[float, List[str]] = {}
    for k in keys:
        buckets.setdefault(round(_implicit_score(k), 2), []).append(k)
    ordered: List[str] = []
    for score in sorted(buckets.keys(), reverse=True):
        band = buckets[score]
        rng.shuffle(band)
        ordered.extend(band)
    keys = ordered
    n = len(keys)
    n_train = int(n * ratios.get("train", 0.8))
    n_val = int(n * ratios.get("val", 0.1))
    train = set(keys[:n_train])
    val = set(keys[n_train : n_train + n_val])

    for r in records:
        k = _stable_group_key(r)
        r["split"] = "train" if k in train else "val" if k in val else "test"
    return records


def split_map(records: List[Dict]) -> Dict[str, List[Dict]]:
    out = {"train": [], "val": [], "test": []}
    for r in records:
        s = r.get("split", "train")
        if s not in out:
            s = "train"
        out[s].append(r)
    return out


def leakage_report(split_rows: Dict[str, List[Dict]]) -> Dict:
    ids = {k: {r.get("review_id") for r in v} for k, v in split_rows.items()}
    overlaps = {}
    keys = ["train", "val", "test"]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = sorted(list(ids[a].intersection(ids[b])))
            overlaps[f"{a}_{b}"] = common[:100]
    text_hashes = {
        k: {hashlib.sha1(r.get("clean_text", "").lower().encode("utf-8")).hexdigest() for r in v}
        for k, v in split_rows.items()
    }
    hash_overlap = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            hash_overlap[f"{a}_{b}"] = len(text_hashes[a].intersection(text_hashes[b]))
    return {"id_overlap_samples": overlaps, "near_exact_text_overlap": hash_overlap}


def _canonical_group_id(record: Dict) -> str:
    for key in ("group_id", "product_id", "review_id", "id"):
        value = str(record.get(key, "")).strip()
        if value:
            return f"{key}:{value.lower()}"
    text = str(record.get("clean_text", "")).strip().lower()
    if text:
        return f"text:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"
    return "unknown:unknown"


def _text_signature(text: str) -> str:
    tokens = [t for t in re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower()).split() if len(t) > 2]
    return " ".join(tokens[:32])


def enforce_split_integrity(records: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
    if not records:
        return records

    split_order = {"train": 0, "val": 1, "test": 2}
    assigned: Dict[str, str] = {}
    grouped: Dict[str, List[Dict]] = {}
    for row in records:
        grouped.setdefault(_canonical_group_id(row), []).append(row)

    for group_key, rows in grouped.items():
        preferred = sorted(
            rows,
            key=lambda r: (
                split_order.get(str(r.get("split", "train")).lower(), 0),
                str(r.get("review_id", "")),
            ),
        )[0]
        split = str(preferred.get("split", "train")).lower()
        split = split if split in split_order else "train"
        assigned[group_key] = split
        for row in rows:
            row["split"] = split

    seen_signatures: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    cleaned: List[Dict] = []
    for split in ["train", "val", "test"]:
        for row in [r for r in records if str(r.get("split", "train")).lower() == split]:
            sig = _text_signature(row.get("clean_text", ""))
            if any(_jaccard(sig, prev) >= similarity_threshold for prev in seen_signatures[split]):
                continue
            other_splits = [k for k in ["train", "val", "test"] if k != split]
            if any(_jaccard(sig, prev) >= similarity_threshold for k in other_splits for prev in seen_signatures[k]):
                continue
            seen_signatures[split].append(sig)
            cleaned.append(row)
    return cleaned


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _group_rows(records: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in records:
        grouped[_stable_group_key(row)].append(row)
    return grouped


def _group_domain(rows: List[Dict]) -> str:
    domain_counts = defaultdict(int)
    for row in rows:
        domain = str(row.get("domain", "general")).strip().lower() or "general"
        domain_counts[domain] += 1
    if not domain_counts:
        return "general"
    return max(domain_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _group_implicit_count(rows: List[Dict]) -> int:
    return sum(1 for row in rows for a in row.get("aspects", []) if str(a.get("aspect_type", "")).lower() == "implicit")


def rebalance_calibration_splits(
    records: List[Dict],
    *,
    seed: int = 42,
    min_implicit_per_split: int = 20,
    preferred_domains: List[str] | None = None,
) -> List[Dict]:
    if not records:
        return records

    preferred_domains = [d.lower() for d in (preferred_domains or ["electronics", "hotel", "telecom", "ecommerce", "delivery"])]
    rng = random.Random(seed)
    grouped = _group_rows(records)

    group_entries = []
    for key, rows in grouped.items():
        split = str(rows[0].get("split", "train")).lower()
        split = split if split in {"train", "val", "test"} else "train"
        domain = _group_domain(rows)
        implicit_count = _group_implicit_count(rows)
        group_entries.append(
            {
                "key": key,
                "rows": rows,
                "split": split,
                "domain": domain,
                "implicit_count": implicit_count,
            }
        )

    def _count(split: str) -> int:
        return sum(g["implicit_count"] for g in group_entries if g["split"] == split)

    def _pick_candidate(target_split: str) -> Dict | None:
        candidates = [g for g in group_entries if g["split"] == "train" and g["implicit_count"] > 0]
        if not candidates:
            return None
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda g: (
                1 if g["domain"] in preferred_domains else 0,
                g["implicit_count"],
                g["domain"],
                g["key"],
            ),
            reverse=True,
        )
        return candidates[0]

    for target_split in ["val", "test"]:
        while _count(target_split) < min_implicit_per_split:
            cand = _pick_candidate(target_split)
            if cand is None:
                break
            cand["split"] = target_split

    out: List[Dict] = []
    for g in group_entries:
        for row in g["rows"]:
            row["split"] = g["split"]
            out.append(row)
    return out
