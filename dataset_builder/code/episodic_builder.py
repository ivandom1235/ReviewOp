from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

STATIC_CONFUSING_LABELS = {
    "battery_life": ["performance", "charging", "heat"],
    "delivery_speed": ["delivery_reliability", "service_speed", "wait_time"],
    "service_speed": ["staff_behavior", "wait_time", "delivery_speed"],
    "food_quality": ["portion_size", "menu_variety", "service_speed"],
    "network_quality": ["performance", "response_time", "customer_support"],
}

TASKS = ["aspect_classification", "implicit_aspect_inference", "aspect_sentiment_classification"]


def _parse_task_mix(raw: str) -> Dict[str, float]:
    out = {k: 0.0 for k in TASKS}
    for part in str(raw or "").split(","):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        if k not in out:
            continue
        try:
            out[k] = max(0.0, float(v.strip()))
        except Exception:
            continue
    total = sum(out.values())
    if total <= 0:
        return {"aspect_classification": 0.4, "implicit_aspect_inference": 0.3, "aspect_sentiment_classification": 0.3}
    return {k: v / total for k, v in out.items()}


def _episode_item(record: Dict, aspect: Dict, *, role: str) -> Dict:
    return {
        "record_id": record["review_id"],
        "text": record["clean_text"],
        "domain": record.get("domain", "general"),
        "aspect_canonical": aspect.get("aspect_canonical"),
        "aspect_type": aspect.get("aspect_type"),
        "sentiment": aspect.get("sentiment"),
        "evidence_text": aspect.get("evidence_text"),
        "source_file": record.get("source_file"),
        "is_augmented": record.get("is_augmented", False),
        "split": record.get("split", "train"),
        "role": role,
    }


def _aspect_rows(records: List[Dict], split: str) -> List[Dict]:
    rows = []
    for r in records:
        if r.get("split") != split:
            continue
        for a in r.get("aspects", []):
            rows.append(
                {
                    "record_id": r["review_id"],
                    "text": r["clean_text"],
                    "domain": r.get("domain", "general"),
                    "aspect_canonical": a.get("aspect_canonical"),
                    "aspect_type": a.get("aspect_type"),
                    "sentiment": a.get("sentiment"),
                    "evidence_text": a.get("evidence_text"),
                    "source_file": r.get("source_file"),
                    "is_augmented": r.get("is_augmented", False),
                    "split": split,
                    "_record": r,
                    "_aspect": a,
                }
            )
    return rows


def _bucket_by_label(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for r in rows:
        lbl = r.get("aspect_canonical")
        if lbl:
            out[lbl].append(r)
    return out


def _data_driven_confusions(rows: List[Dict], top_k: int = 3) -> Dict[str, List[str]]:
    by_record = defaultdict(set)
    for r in rows:
        by_record[r["record_id"]].add(r.get("aspect_canonical"))
    pair_counts = Counter()
    label_counts = Counter()
    for labels in by_record.values():
        labels = sorted([x for x in labels if x])
        for i, a in enumerate(labels):
            label_counts[a] += 1
            for b in labels[i + 1 :]:
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1
    out = defaultdict(list)
    for a in label_counts:
        candidates = sorted(
            [(b, pair_counts[(a, b)]) for b in label_counts if b != a and pair_counts[(a, b)] > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        out[a] = [b for b, _ in candidates[:top_k]]
    return dict(out)


def _hard_negatives(
    labels: List[str],
    label_buckets: Dict[str, List[Dict]],
    hard_negative_k: int,
    strategy: str,
    data_confusions: Dict[str, List[str]],
    rng: random.Random,
) -> List[str]:
    negs = []
    for lbl in labels:
        static = [x for x in STATIC_CONFUSING_LABELS.get(lbl, []) if x in label_buckets and x not in labels]
        dynamic = [x for x in data_confusions.get(lbl, []) if x in label_buckets and x not in labels]
        if strategy == "static":
            candidates = static
        elif strategy == "data_driven":
            candidates = dynamic
        else:
            candidates = list(dict.fromkeys(static + dynamic))
        rng.shuffle(candidates)
        negs.extend(candidates[:hard_negative_k])
    return sorted(list(dict.fromkeys(negs)))


def _records_by_domain(rows: Iterable[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for row in rows:
        out[str(row.get("domain", "general")).lower()].append(row)
    return dict(out)


def _text_signature(text: str) -> str:
    low = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    tokens = [t for t in low.split() if len(t) > 2]
    return " ".join(tokens[:32])


def _jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _select_label_set(
    eligible: List[str],
    label_buckets: Dict[str, List[Dict]],
    n_way: int,
    rng: random.Random,
    task: str,
) -> List[str]:
    if len(eligible) < n_way:
        return []
    weighted = sorted(eligible, key=lambda lbl: (len(label_buckets[lbl]), lbl))
    if task == "implicit_aspect_inference":
        implicit_weighted = [lbl for lbl in weighted if any(x.get("aspect_type") == "implicit" for x in label_buckets[lbl])]
        if len(implicit_weighted) >= n_way:
            weighted = implicit_weighted
        elif len(implicit_weighted) > 0:
            weighted = implicit_weighted + [lbl for lbl in weighted if lbl not in implicit_weighted]
    if len(weighted) < n_way:
        weighted = list(dict.fromkeys(eligible))
    if len(weighted) <= n_way:
        return rng.sample(weighted, min(n_way, len(weighted)))
    # Prefer labels with a dense bucket, but inject confusion-neighbors when possible.
    base = [weighted[0]]
    candidates = [lbl for lbl in weighted[1:] if lbl not in base]
    rng.shuffle(candidates)
    while len(base) < n_way and candidates:
        base.append(candidates.pop(0))
    if len(base) < n_way:
        remaining = [lbl for lbl in eligible if lbl not in base]
        rng.shuffle(remaining)
        base.extend(remaining[: n_way - len(base)])
    return base[:n_way]


def _pick_examples(
    bucket: List[Dict],
    k_needed: int,
    rng: random.Random,
    *,
    exclude_ids: set[str] | None = None,
    aspect_type: str | None = None,
    sentiment_preference: Tuple[str, ...] | None = None,
    domain_preference: set[str] | None = None,
) -> List[Dict]:
    exclude_ids = exclude_ids or set()
    pool = [x for x in bucket if x["record_id"] not in exclude_ids]
    if aspect_type:
        filtered = [x for x in pool if x.get("aspect_type") == aspect_type]
        if filtered:
            pool = filtered
    if sentiment_preference:
        filtered = [x for x in pool if str(x.get("sentiment", "")).lower() in sentiment_preference]
        if filtered:
            pool = filtered
    if domain_preference:
        filtered = [x for x in pool if str(x.get("domain", "general")).lower() in domain_preference]
        if filtered:
            pool = filtered
    rng.shuffle(pool)
    picked = []
    seen_records = set()
    for ex in pool:
        if ex["record_id"] in seen_records:
            continue
        seen_records.add(ex["record_id"])
        picked.append(ex)
        if len(picked) >= k_needed:
            break
    return picked


def _pick_non_leaky_examples(
    bucket: List[Dict],
    k_needed: int,
    rng: random.Random,
    *,
    exclude_ids: set[str] | None = None,
    exclude_texts: List[str] | None = None,
    aspect_type: str | None = None,
    sentiment_preference: Tuple[str, ...] | None = None,
    domain_preference: set[str] | None = None,
    max_similarity: float = 0.82,
) -> List[Dict]:
    pool = _pick_examples(
        bucket,
        len(bucket),
        rng,
        exclude_ids=exclude_ids,
        aspect_type=aspect_type,
        sentiment_preference=sentiment_preference,
        domain_preference=domain_preference,
    )
    exclude_texts = [_text_signature(x) for x in (exclude_texts or [])]
    picked: List[Dict] = []
    for ex in pool:
        sig = _text_signature(ex.get("text", ""))
        if any(_jaccard(sig, t) >= max_similarity for t in exclude_texts):
            continue
        if any(_jaccard(sig, _text_signature(p.get("text", ""))) >= max_similarity for p in picked):
            continue
        picked.append(ex)
        if len(picked) >= k_needed:
            break
    return picked


def _domain_coherent_labels(label_pool: List[str], by_label: Dict[str, List[Dict]], n_way: int, rng: random.Random, allow_mixed: bool) -> List[str]:
    if allow_mixed:
        return _select_label_set(label_pool, by_label, n_way, rng, task="aspect_classification")
    domain_counts = Counter()
    for lbl in label_pool:
        for d in {str(x.get("domain", "general")).lower() for x in by_label[lbl]}:
            domain_counts[d] += 1
    if not domain_counts:
        return _select_label_set(label_pool, by_label, n_way, rng, task="aspect_classification")
    target_domain = domain_counts.most_common(1)[0][0]
    coherent = [lbl for lbl in label_pool if any(str(x.get("domain", "general")).lower() == target_domain for x in by_label[lbl])]
    return _select_label_set(coherent or label_pool, by_label, n_way, rng, task="aspect_classification")


def _validate_episode(ep: Dict, enforce_labels_field: bool, balance_tolerance: float, k_shot: int, q_query: int) -> bool:
    labels = ep.get("labels", [])
    if enforce_labels_field and (not labels or not isinstance(labels, list)):
        return False
    if not ep.get("support_set") or not ep.get("query_set"):
        return False

    support_ids = {x["record_id"] for x in ep["support_set"]}
    query_ids = {x["record_id"] for x in ep["query_set"]}
    if support_ids.intersection(query_ids):
        return False

    per_label_support = defaultdict(int)
    per_label_query = defaultdict(int)
    for x in ep["support_set"]:
        per_label_support[x["aspect_canonical"]] += 1
    for x in ep["query_set"]:
        per_label_query[x["aspect_canonical"]] += 1

    if set(per_label_support.keys()) != set(labels) or set(per_label_query.keys()) != set(labels):
        return False

    for lbl in labels:
        if abs(per_label_support[lbl] - k_shot) > balance_tolerance:
            return False
        if abs(per_label_query[lbl] - q_query) > balance_tolerance:
            return False
    return True


def _episode_domain(support_set: List[Dict], query_set: List[Dict]) -> str:
    domains = {str(x.get("domain", "general")).lower() for x in support_set + query_set}
    if len(domains) <= 1:
        return next(iter(domains), "general")
    return "mixed"


def _task_counts(total: int, task_mix: Dict[str, float]) -> Dict[str, int]:
    counts = {t: max(1, int(round(total * task_mix.get(t, 0.0)))) for t in TASKS}
    # Keep a bounded total episode budget for faster default runs.
    cap = max(6, total)
    running = sum(counts.values())
    if running > cap:
        scale = cap / running
        counts = {t: max(1, int(round(v * scale))) for t, v in counts.items()}
    return counts


def _build_episode(
    *,
    split: str,
    task: str,
    index: int,
    labels: List[str],
    by_label: Dict[str, List[Dict]],
    hard_negative_k: int,
    hard_negative_strategy: str,
    data_confusions: Dict[str, List[str]],
    k_shot: int,
    q_query: int,
    implicit_query_only: bool,
    cross_domain: bool,
    cross_domain_min_domains: int,
    enforce_labels_field: bool,
    balance_tolerance: float,
    rng: random.Random,
    train_domains: set[str],
) -> Dict | None:
    support_set: List[Dict] = []
    query_set: List[Dict] = []
    used_support_ids: set[str] = set()
    used_query_ids: set[str] = set()

    preferred_query_domain = None
    if cross_domain and split in {"val", "test"} and train_domains:
        preferred_query_domain = {d for d in train_domains if d}

    for lbl in labels:
        bucket = list(by_label[lbl])
        if cross_domain and split in {"val", "test"} and train_domains:
            cross_bucket = [x for x in bucket if str(x.get("domain", "general")).lower() not in train_domains]
            if cross_bucket:
                bucket = cross_bucket
        rng.shuffle(bucket)

        support_aspect_type = None
        query_aspect_type = None
        if task == "implicit_aspect_inference":
            query_aspect_type = "implicit"
            if not any(x.get("aspect_type") == "implicit" for x in bucket):
                return None
        if task == "aspect_sentiment_classification":
            query_aspect_type = None

        support = _pick_non_leaky_examples(bucket, k_shot, rng, exclude_ids=used_support_ids, aspect_type=support_aspect_type)
        if len(support) < k_shot:
            return None
        support_ids = {x["record_id"] for x in support}
        used_support_ids.update(support_ids)
        support_texts = [x.get("text", "") for x in support]

        query_candidates = [x for x in bucket if x["record_id"] not in used_support_ids and x["record_id"] not in used_query_ids]
        if task == "implicit_aspect_inference" and implicit_query_only:
            implicit_candidates = [x for x in query_candidates if x.get("aspect_type") == "implicit"]
            if implicit_candidates:
                query_candidates = implicit_candidates
        if task == "aspect_sentiment_classification":
            sentiments = [str(x.get("sentiment", "")).lower() for x in query_candidates]
            if len(set(sentiments)) < 2:
                sentiment_pref = None
            else:
                sentiment_pref = ("positive", "negative", "neutral")
        else:
            sentiment_pref = None
        if preferred_query_domain:
            domain_candidates = [x for x in query_candidates if str(x.get("domain", "general")).lower() not in train_domains]
            if domain_candidates:
                query_candidates = domain_candidates
        query = _pick_non_leaky_examples(
            query_candidates,
            q_query,
            rng,
            exclude_ids=used_query_ids,
            exclude_texts=support_texts,
            aspect_type=query_aspect_type,
            sentiment_preference=sentiment_pref,
        )
        if len(query) < q_query:
            return None
        query_ids = {x["record_id"] for x in query}
        used_query_ids.update(query_ids)

        support_set.extend([_episode_item(ex["_record"], ex["_aspect"], role="support") for ex in support])
        query_set.extend([_episode_item(ex["_record"], ex["_aspect"], role="query") for ex in query])

    ep = {
        "episode_id": f"ep_{split}_{task}_{index+1:05d}",
        "task_type": task,
        "domain": _episode_domain(support_set, query_set),
        "n_way": len(labels),
        "k_shot": k_shot,
        "q_query": q_query,
        "labels": labels,
        "hard_negative_labels": _hard_negatives(labels, by_label, hard_negative_k, hard_negative_strategy, data_confusions, rng),
        "support_set": support_set,
        "query_set": query_set,
        "is_augmented_episode": any(x.get("is_augmented", False) for x in support_set + query_set),
        "split": split,
    }
    if cross_domain and split in {"val", "test"}:
        dcount = len({str(x.get("domain", "general")).lower() for x in support_set + query_set})
        if dcount < max(1, cross_domain_min_domains):
            return None
    if _validate_episode(ep, enforce_labels_field=enforce_labels_field, balance_tolerance=balance_tolerance, k_shot=k_shot, q_query=q_query):
        return ep
    return None


def build_episodes(
    records: List[Dict],
    n_way: int,
    k_shot: int,
    q_query: int,
    hard_negative_k: int,
    hard_negative_strategy: str,
    episode_task_mix: str,
    implicit_query_only: bool,
    cross_domain: bool,
    cross_domain_min_domains: int,
    enforce_labels_field: bool,
    balance_tolerance: float,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    episodes: List[Dict] = []
    task_mix = _parse_task_mix(episode_task_mix)
    train_domains = {str(r.get("domain", "general")).lower() for r in records if r.get("split") == "train"}

    for split in ["train", "val", "test"]:
        rows = _aspect_rows(records, split)
        if not rows:
            continue
        by_label = _bucket_by_label(rows)
        data_confusions = _data_driven_confusions(rows)
        eligible = [
            lbl
            for lbl, bucket in by_label.items()
            if len(bucket) >= (k_shot + q_query)
            and len({str(x.get("record_id")) for x in bucket}) >= (k_shot + q_query)
        ]
        if len(eligible) < n_way:
            continue

        label_pool = sorted(eligible, key=lambda lbl: (-len(by_label[lbl]), lbl))
        # Runtime optimization: fewer episodes still keep split/task diversity.
        desired_total = min(180, max(36, len(label_pool) * 6))
        task_counts = _task_counts(desired_total, task_mix)
        allow_mixed = bool(cross_domain)

        for task in TASKS:
            for i in range(task_counts[task]):
                labels = _domain_coherent_labels(label_pool, by_label, n_way, rng, allow_mixed=allow_mixed)
                if len(labels) != n_way:
                    continue
                if n_way < 2:
                    continue
                if task == "implicit_aspect_inference":
                    labels = [lbl for lbl in labels if any(x.get("aspect_type") == "implicit" for x in by_label[lbl])] or labels
                ep = _build_episode(
                    split=split,
                    task=task,
                    index=i,
                    labels=labels,
                    by_label=by_label,
                    hard_negative_k=hard_negative_k,
                    hard_negative_strategy=hard_negative_strategy,
                    data_confusions=data_confusions,
                    k_shot=k_shot,
                    q_query=q_query,
                    implicit_query_only=implicit_query_only,
                    cross_domain=cross_domain,
                    cross_domain_min_domains=cross_domain_min_domains,
                    enforce_labels_field=enforce_labels_field,
                    balance_tolerance=balance_tolerance,
                    rng=rng,
                    train_domains=train_domains,
                )
                if ep is not None:
                    episodes.append(ep)

    return episodes


def build_episodic_rows(review_rows: List[Dict]) -> List[Dict]:
    """Backward-compatible converter used by build_dataset.py."""
    out: List[Dict] = []
    for row in review_rows:
        labels = row.get("labels", [])
        for idx, label in enumerate(labels, start=1):
            aspect = str(label.get("aspect", "")).strip()
            if not aspect:
                continue
            out.append(
                {
                    "example_id": f"{row['id']}_e{idx}",
                    "parent_review_id": row["id"],
                    "review_text": row.get("review_text", ""),
                    "evidence_sentence": label.get("evidence_sentence", ""),
                    "domain": row.get("domain", "generic"),
                    "aspect": aspect,
                    "implicit_aspect": label.get("implicit_aspect", aspect),
                    "sentiment": label.get("sentiment", "neutral"),
                    "label_type": label.get("type", "explicit"),
                    "split": row.get("split", "train"),
                    "source": row.get("source", ""),
                }
            )
    return out
