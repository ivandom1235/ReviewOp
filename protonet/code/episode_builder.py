from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List
import numpy as np

try:
    from .config import ProtonetConfig
    from .dataset_reader import write_jsonl
    from .progress import announce, track
except ImportError:
    from config import ProtonetConfig
    from dataset_reader import write_jsonl
    from progress import announce, track


def build_joint_label(row: Dict[str, Any], separator: str = "__") -> str:
    aspect = str(row.get("aspect") or row.get("implicit_aspect") or "unknown").strip()
    sentiment = str(row.get("sentiment") or "neutral").strip().lower() or "neutral"
    return f"{aspect}{separator}{sentiment}"


def _stable_shuffle(rows: List[Dict[str, Any]], seed: int, salt: str) -> List[Dict[str, Any]]:
    keyed = []
    for row in rows:
        ident = str(row.get("example_id") or row.get("record_id") or row.get("parent_review_id") or "")
        digest = hashlib.sha1(f"{seed}|{salt}|{ident}".encode("utf-8")).hexdigest()
        keyed.append((digest, row))
    keyed.sort(key=lambda item: item[0])
    return [row for _, row in keyed]


def is_prebuilt_episode_row(row: Dict[str, Any]) -> bool:
    return "support_set" in row and "query_set" in row


def validate_episode_row(episode: Dict[str, Any], cfg: ProtonetConfig) -> None:
    support = episode.get("support_set", [])
    query = episode.get("query_set", [])
    if not support or not query:
        raise ValueError(f"Episode {episode.get('episode_id', 'unknown')} is missing support/query rows")
    support_ids = {str(item.get("parent_review_id") or item.get("record_id") or item.get("example_id")) for item in support}
    query_ids = {str(item.get("parent_review_id") or item.get("record_id") or item.get("example_id")) for item in query}
    overlap = support_ids.intersection(query_ids)
    if overlap:
        raise ValueError(f"Episode {episode.get('episode_id', 'unknown')} leaks review ids: {sorted(overlap)[:3]}")
    labels = episode.get("labels", [])
    if len(labels) != episode.get("n_way"):
        raise ValueError(f"Episode {episode.get('episode_id', 'unknown')} has mismatched labels and n_way")
    if episode.get("k_shot") != cfg.k_shot or episode.get("q_query") != cfg.q_query:
        raise ValueError(f"Episode {episode.get('episode_id', 'unknown')} does not match configured shot sizes")


def _episode_cache_path(cfg: ProtonetConfig, split: str) -> Path:
    return _episode_cache_path_with_protocol(cfg, split, protocol=None)


def _episode_cache_path_with_protocol(cfg: ProtonetConfig, split: str, protocol: str | None) -> Path:
    protocol_suffix = f"_p{protocol}" if protocol else ""
    return cfg.episode_cache_dir / (
        f"{cfg.input_type}_{split}{protocol_suffix}_n{cfg.n_way}_k{cfg.k_shot}_q{cfg.q_query}_"
        f"seed{cfg.seed}_train{cfg.max_train_episodes}_eval{cfg.max_eval_episodes}.jsonl"
    )


def _load_cached_episodes(cfg: ProtonetConfig, split: str, protocol: str | None = None) -> List[Dict[str, Any]] | None:
    if cfg.force_rebuild_episodes:
        return None
    path = _episode_cache_path_with_protocol(cfg, split, protocol)
    if not path.exists():
        return None
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if rows:
        return rows
    return None


def _filter_rows_by_protocol(rows: List[Dict[str, Any]], split: str, protocol: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        payload = row.get("split_protocol") if isinstance(row.get("split_protocol"), dict) else {}
        protocol_split = str(payload.get(protocol) or row.get("split") or split)
        if protocol_split == split:
            out.append(row)
    return out


def _dedupe_by_parent(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        parent = str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id"))
        if parent in seen:
            continue
        seen.add(parent)
        out.append(row)
    return out


def _group_examples(rows: List[Dict[str, Any]], cfg: ProtonetConfig) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[build_joint_label(row, cfg.joint_label_separator)].append(row)
    return grouped


def _eligible_labels(grouped: Dict[str, List[Dict[str, Any]]], cfg: ProtonetConfig) -> List[str]:
    labels: List[str] = []
    needed = cfg.k_shot + cfg.q_query
    for label, rows in grouped.items():
        unique_parents = {str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id")) for row in rows}
        if len(unique_parents) >= max(needed, cfg.min_examples_per_label):
            labels.append(label)
    return sorted(labels)


def _episode_row_from_example(row: Dict[str, Any], role: str, cfg: ProtonetConfig) -> Dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "parent_review_id": row.get("parent_review_id"),
        "review_text": row.get("review_text"),
        "evidence_sentence": row.get("evidence_sentence") or row.get("review_text"),
        "evidence_fallback_used": bool(row.get("evidence_fallback_used", False)),
        "domain": row.get("domain", "unknown"),
        "domain_family": row.get("domain_family", ""),
        "group_id": row.get("group_id", ""),
        "aspect": row.get("aspect") or row.get("implicit_aspect"),
        "sentiment": str(row.get("sentiment") or "neutral").lower(),
        "label_type": row.get("label_type", "explicit"),
        "confidence": float(row.get("confidence", 1.0)),
        "hardness_tier": str(row.get("hardness_tier") or "H0").upper(),
        "annotation_source": row.get("annotation_source", "unknown"),
        "joint_label": build_joint_label(row, cfg.joint_label_separator),
        "role": role,
        "gold_joint_labels": list(row.get("gold_joint_labels") or []),
        "gold_interpretations": list(row.get("gold_interpretations") or []),
        "abstain_acceptable": bool(row.get("abstain_acceptable", False)),
        "ambiguity_type": row.get("ambiguity_type"),
        "benchmark_ambiguity_score": float(row.get("benchmark_ambiguity_score", 0.0)),
        "novel_acceptable": bool(row.get("novel_acceptable", False)),
        "novel_cluster_id": row.get("novel_cluster_id"),
        "novel_alias": row.get("novel_alias"),
        "novel_evidence_text": row.get("novel_evidence_text"),
        "split_protocol": row.get("split_protocol") or {},
    }


def _select_rows(rows: List[Dict[str, Any]], count: int, seed: int, salt: str) -> List[Dict[str, Any]]:
    shuffled = _stable_shuffle(rows, seed=seed, salt=salt)
    deduped = _dedupe_by_parent(shuffled)
    return deduped[:count]


def _select_rows_excluding(
    rows: List[Dict[str, Any]],
    count: int,
    *,
    excluded_parent_ids: set[str],
    seed: int,
    salt: str,
) -> List[Dict[str, Any]]:
    filtered = [
        row for row in rows
        if str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id")) not in excluded_parent_ids
    ]
    return _select_rows(filtered, count, seed, salt)


def _build_episodes_for_split(split: str, rows: List[Dict[str, Any]], cfg: ProtonetConfig) -> List[Dict[str, Any]]:
    grouped = _group_examples(rows, cfg)
    labels = _eligible_labels(grouped, cfg)
    if len(labels) < cfg.n_way:
        raise ValueError(
            f"Split {split} does not have enough eligible labels for n_way={cfg.n_way}. "
            f"Found {len(labels)} eligible labels."
        )

    max_episodes = cfg.max_train_episodes if split == "train" else cfg.max_eval_episodes
    episodes: List[Dict[str, Any]] = []
    split_seed = sum(ord(ch) for ch in split)
    rng = random.Random(cfg.seed + split_seed)
    max_attempts = max(max_episodes * 12, 24)
    attempt = 0
    label_weights = _compute_label_weights(grouped, labels)
    while len(episodes) < max_episodes and attempt < max_attempts:
        attempt += 1
        chosen_labels = _sample_labels_weighted(labels, label_weights, cfg.n_way, rng)
        support_set: List[Dict[str, Any]] = []
        query_set: List[Dict[str, Any]] = []
        support_parent_ids: set[str] = set()
        query_parent_ids: set[str] = set()
        can_build = True
        for label in chosen_labels:
            bucket = grouped[label]
            support_examples = _select_rows_excluding(
                _weighted_rows(bucket),
                cfg.k_shot,
                excluded_parent_ids=query_parent_ids,
                seed=cfg.seed + attempt,
                salt=f"{split}:{label}:support",
            )
            if len(support_examples) < cfg.k_shot:
                can_build = False
                break
            support_ids_for_label = {
                str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id"))
                for row in support_examples
            }
            query_examples = _select_rows_excluding(
                _weighted_rows(bucket),
                cfg.q_query,
                excluded_parent_ids=support_parent_ids.union(support_ids_for_label),
                seed=cfg.seed + attempt,
                salt=f"{split}:{label}:query",
            )
            if len(query_examples) < cfg.q_query:
                can_build = False
                break
            support_parent_ids.update(support_ids_for_label)
            query_parent_ids.update(
                str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id"))
                for row in query_examples
            )
            support_set.extend(_episode_row_from_example(row, "support", cfg) for row in support_examples)
            query_set.extend(_episode_row_from_example(row, "query", cfg) for row in query_examples)
        if not can_build:
            continue
        episode = {
            "episode_id": f"{split}_ep_{len(episodes) + 1:04d}",
            "split": split,
            "n_way": cfg.n_way,
            "k_shot": cfg.k_shot,
            "q_query": cfg.q_query,
            "labels": sorted(chosen_labels),
            "support_set": support_set,
            "query_set": query_set,
            "domain": "mixed" if len({item["domain"] for item in support_set + query_set}) > 1 else support_set[0]["domain"],
        }
        validate_episode_row(episode, cfg)
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"No valid episodes were built for split {split}")
    return episodes


def _hardness_score(row: Dict[str, Any]) -> float:
    tier = str(row.get("hardness_tier") or "H0").strip().upper()
    return {"H0": 0.0, "H1": 0.34, "H2": 0.67, "H3": 1.0}.get(tier, 0.0)


def _compute_label_weights(grouped: Dict[str, List[Dict[str, Any]]], labels: List[str]) -> Dict[str, float]:
    max_domains = max(
        1,
        max(len({str(row.get("domain") or "unknown") for row in rows}) for rows in grouped.values()),
    )
    out: Dict[str, float] = {}
    for label in labels:
        rows = grouped.get(label, [])
        unique_parents = len({str(row.get("parent_review_id") or row.get("record_id") or row.get("example_id")) for row in rows})
        rarity = 1.0 / max(1, unique_parents)
        hard = float(np.mean([_hardness_score(row) for row in rows])) if rows else 0.0
        diverse = len({str(row.get("domain") or "unknown") for row in rows}) / max_domains if rows else 0.0
        out[label] = max(1e-6, 0.55 * rarity + 0.30 * hard + 0.15 * diverse)
    return out


def _sample_labels_weighted(labels: List[str], weights: Dict[str, float], n_way: int, rng: random.Random) -> List[str]:
    chosen: List[str] = []
    available = list(labels)
    while available and len(chosen) < n_way:
        mass = [float(weights.get(label, 1.0)) for label in available]
        pick = rng.choices(available, weights=mass, k=1)[0]
        chosen.append(pick)
        available.remove(pick)
    if len(chosen) < n_way:
        return sorted(labels)[:n_way]
    return chosen


def _weighted_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def row_score(row: Dict[str, Any]) -> tuple[float, str]:
        grounded = 1.0 if not bool(row.get("evidence_fallback_used", False)) else 0.0
        hard = _hardness_score(row)
        multi = 1.0 if len(list(row.get("gold_joint_labels") or [])) >= 2 else 0.0
        boundary = 1.0 if bool(row.get("abstain_acceptable", False)) else 0.0
        score = 0.35 * grounded + 0.30 * hard + 0.20 * multi + 0.15 * boundary
        ident = str(row.get("example_id") or row.get("parent_review_id") or "")
        return (-score, ident)
    return sorted(rows, key=row_score)


def build_or_load_episode_sets(
    rows_by_split: Dict[str, List[Dict[str, Any]]],
    cfg: ProtonetConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    episodes_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        cached = _load_cached_episodes(cfg, split)
        if cached is not None:
            try:
                for episode in cached:
                    validate_episode_row(episode, cfg)
            except ValueError:
                cached = None
            else:
                episodes_by_split[split] = cached
                announce(f"Loaded cached {split} episodes from {_episode_cache_path(cfg, split)}")
                continue

        if rows and is_prebuilt_episode_row(rows[0]):
            for episode in rows:
                validate_episode_row(episode, cfg)
            episodes = rows
        else:
            episodes = _build_episodes_for_split(split, rows, cfg)
        episodes_by_split[split] = episodes
        write_jsonl(_episode_cache_path(cfg, split), track(episodes, total=len(episodes), desc=f"save:{split}", enabled=cfg.progress_enabled))

    if cfg.protocol_eval_enabled:
        for protocol in cfg.protocol_eval_splits:
            for split in ("val", "test"):
                protocol_rows = _filter_rows_by_protocol(rows_by_split.get(split, []), split, protocol)
                if not protocol_rows:
                    continue
                protocol_key = f"{split}__{protocol}"
                cached_path = _episode_cache_path_with_protocol(cfg, split, protocol)
                cached = _load_cached_episodes(cfg, split, protocol)
                if cached is not None:
                    try:
                        for episode in cached:
                            validate_episode_row(episode, cfg)
                    except ValueError:
                        cached = None
                    else:
                        episodes_by_split[protocol_key] = cached
                        announce(f"Loaded cached {protocol_key} episodes from {cached_path}")
                        continue
                try:
                    protocol_episodes = _build_episodes_for_split(split, protocol_rows, cfg)
                except ValueError as exc:
                    announce(f"Skipping protocol episode cache {protocol_key}: {exc}")
                    continue
                episodes_by_split[protocol_key] = protocol_episodes
                write_jsonl(cached_path, track(protocol_episodes, total=len(protocol_episodes), desc=f"save:{protocol_key}", enabled=cfg.progress_enabled))
    return episodes_by_split
