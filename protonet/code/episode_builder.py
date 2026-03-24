from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List

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
    return cfg.episode_cache_dir / (
        f"{cfg.input_type}_{split}_n{cfg.n_way}_k{cfg.k_shot}_q{cfg.q_query}_"
        f"seed{cfg.seed}_train{cfg.max_train_episodes}_eval{cfg.max_eval_episodes}.jsonl"
    )


def _load_cached_episodes(cfg: ProtonetConfig, split: str) -> List[Dict[str, Any]] | None:
    if cfg.force_rebuild_episodes:
        return None
    path = _episode_cache_path(cfg, split)
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
        "domain": row.get("domain", "unknown"),
        "aspect": row.get("aspect") or row.get("implicit_aspect"),
        "sentiment": str(row.get("sentiment") or "neutral").lower(),
        "label_type": row.get("label_type", "explicit"),
        "confidence": float(row.get("confidence", 1.0)),
        "joint_label": build_joint_label(row, cfg.joint_label_separator),
        "role": role,
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
    while len(episodes) < max_episodes and attempt < max_attempts:
        attempt += 1
        chosen_labels = rng.sample(labels, cfg.n_way)
        support_set: List[Dict[str, Any]] = []
        query_set: List[Dict[str, Any]] = []
        support_parent_ids: set[str] = set()
        query_parent_ids: set[str] = set()
        can_build = True
        for label in chosen_labels:
            bucket = grouped[label]
            support_examples = _select_rows_excluding(
                bucket,
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
                bucket,
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
    return episodes_by_split
