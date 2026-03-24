from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from typing import Any, Dict, List

from aspect_infer import build_evidence_metadata, choose_evidence_sentence, collect_labels_for_row, configure_inference, force_canonical_aspect, infer_aspect_family, summarize_aspects
from config import BuilderConfig
from domain_infer import infer_domain
from episodic_builder import build_episodic_rows
from mappings import (
    BROAD_CANONICAL_ASPECTS,
    CANONICAL_ASPECTS,
    CANONICAL_EXPORT_FALLBACK,
    CANONICAL_EXPORT_REDISTRIBUTION,
    DOMAIN_DETAIL_HINTS,
    EXPORT_CANONICAL_ASPECTS,
    SURFACE_STOPWORDS,
)
from quality_diagnostics import (
    build_data_quality_report,
    build_episode_readiness_report,
    build_label_issue_candidates,
)
from senticnet_utils import configure_senticnet, senticnet_status, senticnet_vote
from schema_detect import detect_schema, load_file_rows
from splitter import assign_splits, leakage_ids, split_rows, apply_domain_mixing
from tqdm import tqdm
from utils import normalize_text, stable_hash, write_json, write_jsonl
from validators import aspect_frequency, few_shot_warnings, validate_jsonl, validate_review_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPEN_ASPECT_RESCUE_HINTS = {
    "value": {"price", "pricing", "cost", "worth", "cheap", "expensive", "deal", "bill", "money", "value"},
    "support_quality": {"service", "staff", "server", "waiter", "waitress", "host", "owner", "manager", "bartender", "support", "helpful"},
    "delivery_logistics": {"delivery", "arrived", "arrival", "shipping", "package", "reservation", "reservations", "table", "seating"},
    "performance": {"slow", "fast", "speed", "responsive", "waited", "minutes", "hour", "lag", "battery", "charge", "hot", "fan"},
    "product_quality": {"taste", "fresh", "flavor", "bland", "salty", "delicious", "portion", "meal", "dish", "pizza", "sushi", "bagel", "dessert", "drink", "food", "screen", "display", "speaker", "sound", "build", "quality"},
    "aesthetics": {"decor", "design", "presentation", "plating", "ambience", "atmosphere", "music", "appearance", "look", "beautiful"},
    "experience": {"place", "space", "noise", "crowded", "ambience", "atmosphere", "music", "comfort"},
    "usability": {"easy", "easier", "difficult", "intuitive", "confusing", "keyboard", "touchpad", "mouse", "navigation", "use"},
    "reliability": {"broken", "broke", "reliable", "stable", "crash", "durable", "warranty", "lifespan"},
    "compatibility": {"windows", "software", "program", "programs", "application", "applications", "os", "driver"},
}

DOMAIN_OPEN_FALLBACK = {
    "food": "product_quality",
    "hospitality": "experience",
    "electronics": "product_quality",
    "software": "usability",
}


def _detail_alias_to_family() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for domain_map in CANONICAL_ASPECTS.values():
        for family, aliases in domain_map.items():
            for alias in aliases:
                out[str(alias).strip().lower()] = family
    return out


DETAIL_ALIAS_TO_FAMILY = _detail_alias_to_family()


def _approved_detail_candidates(domain: str, aspect: str) -> set[str]:
    domain = str(domain or "generic").strip().lower() or "generic"
    if aspect.startswith("other_"):
        return {detail for detail in DOMAIN_DETAIL_HINTS.get(domain, {}) if detail in EXPORT_CANONICAL_ASPECTS}
    candidates = CANONICAL_EXPORT_REDISTRIBUTION.get(domain, {}).get(aspect, set())
    return {detail for detail in candidates if detail in EXPORT_CANONICAL_ASPECTS}


def _select_detail_canonical(
    *,
    domain: str,
    current_aspect: str,
    metadata: Dict[str, Any],
    evidence_sentence: str,
    review_text: str,
    implicit_aspect: str = "",
) -> str:
    candidates = _approved_detail_candidates(domain, current_aspect)
    if not candidates:
        return ""

    raw_aspect = str(metadata.get("raw_aspect", "")).strip()
    surface = str(metadata.get("aspect_surface", "")).strip()
    matched_symptom = str(metadata.get("matched_symptom", "")).strip()
    sentic = senticnet_vote(" ".join([raw_aspect, surface, matched_symptom, evidence_sentence, review_text]).strip(), domain=domain)
    tokens = _tokenize_hint_text(raw_aspect, surface, matched_symptom, evidence_sentence, review_text, implicit_aspect)
    best_score = 0.0
    best_detail = ""
    for detail in sorted(candidates):
        hints = DOMAIN_DETAIL_HINTS.get(domain, {}).get(detail, [])
        detail_tokens = _tokenize_hint_text(detail, *hints)
        score = float(len(tokens.intersection(detail_tokens)))
        score += float(sentic.get("aspect_scores", {}).get(detail, 0.0))
        if raw_aspect and normalize_text(raw_aspect).lower().replace(" ", "_") == detail:
            score += 2.4
        if surface and normalize_text(surface).lower().replace(" ", "_") == detail:
            score += 2.0
        if implicit_aspect and normalize_text(implicit_aspect).lower().replace(" ", "_") == detail:
            score += 1.8
        if detail in tokens:
            score += 1.25
        if score > best_score:
            best_score = score
            best_detail = detail
    return best_detail if best_score >= 2.15 else ""


def _canonicalize_export_labels(review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in review_rows:
        domain = str(row.get("domain", "generic")).strip().lower() or "generic"
        review_text = str(row.get("review_text", "")).strip()
        for label in row.get("labels", []):
            metadata = dict(label.get("metadata", {}))
            original_aspect = str(label.get("aspect", "")).strip()
            implicit_surface = str(label.get("implicit_aspect", "")).strip()
            canonical = force_canonical_aspect(original_aspect or implicit_surface or metadata.get("raw_aspect", ""), domain)

            detail_candidate = _select_detail_canonical(
                domain=domain,
                current_aspect=canonical or original_aspect,
                metadata=metadata,
                evidence_sentence=str(label.get("evidence_sentence", "")),
                review_text=review_text,
                implicit_aspect=implicit_surface,
            )
            if detail_candidate:
                if canonical != detail_candidate:
                    metadata["redistributed_from"] = canonical or original_aspect
                canonical = detail_candidate

            if original_aspect and original_aspect != canonical:
                metadata.setdefault("canonicalized_from", original_aspect)

            surface_detail = str(metadata.get("aspect_surface") or implicit_surface or metadata.get("raw_aspect") or "").strip()
            if surface_detail:
                surface_key = normalize_text(surface_detail).lower().replace(" ", "_")
                if surface_key in EXPORT_CANONICAL_ASPECTS:
                    metadata["aspect_surface"] = surface_key
                elif surface_key in SURFACE_STOPWORDS:
                    metadata["aspect_surface"] = ""
                else:
                    metadata["aspect_surface"] = surface_key
            else:
                metadata.setdefault("aspect_surface", "")

            metadata["aspect_family"] = infer_aspect_family(canonical, domain) or canonical
            label["aspect"] = canonical or CANONICAL_EXPORT_FALLBACK.get(domain, "product_quality")
            label["implicit_aspect"] = metadata.get("aspect_surface") or label["aspect"]
            label["metadata"] = metadata
    return review_rows


def _shorten_augmentation_evidence(text: str, aspect: str) -> str:
    candidate = choose_evidence_sentence(text, aspect, "")
    if candidate and normalize_text(candidate) != normalize_text(text):
        return candidate
    clauses = [part.strip() for part in re.split(r"[,;:]|\bbut\b|\bbecause\b|\bwhile\b|\band\b", str(text or ""), flags=re.IGNORECASE) if part.strip()]
    if not clauses:
        return str(text or "")
    aspect_hint = aspect.replace("_", " ").lower()
    for clause in sorted(clauses, key=len):
        if aspect_hint and aspect_hint in clause.lower():
            return clause
    return min(clauses, key=len)


def _resolve_project_path(raw: Path) -> Path:
    if raw.is_absolute():
        return raw
    if len(raw.parts) >= 2 and raw.parts[0] == "dataset_builder":
        return PROJECT_ROOT / Path(*raw.parts[1:])
    return PROJECT_ROOT / raw


def list_input_files(input_dir: Path) -> List[Path]:
    supported = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".xml", ".gz"}
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
    return sorted(files)


def extract_aspect_values(row: Dict[str, Any], aspect_cols: List[str]) -> List[str]:
    values: List[str] = []
    for col in aspect_cols:
        raw = row.get(col, "")
        if isinstance(raw, list):
            values.extend(normalize_text(x) for x in raw)
        elif isinstance(raw, str):
            if "," in raw:
                values.extend(normalize_text(x) for x in raw.split(","))
            elif ";" in raw:
                values.extend(normalize_text(x) for x in raw.split(";"))
            else:
                values.append(normalize_text(raw))
        elif raw:
            values.append(normalize_text(raw))
    return [v for v in values if v]


def ensure_output_dirs(output_dir: Path) -> None:
    for d in ["reviewlevel", "episodic"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)


def compact_open_aspects(review_rows: List[Dict[str, Any]], min_count: int = 5, *, domain_agnostic_mode: str = "auto") -> List[Dict[str, Any]]:
    freq = Counter()
    for row in review_rows:
        for lab in row.get("labels", []):
            freq[str(lab.get("aspect", ""))] += 1

    for row in review_rows:
        domain = str(row.get("domain", "generic")).strip().lower() or "generic"
        review_text = str(row.get("review_text", "")).strip()
        merged = {}
        for lab in row.get("labels", []):
            aspect = str(lab.get("aspect", "")).strip()
            mode = str(lab.get("metadata", {}).get("mapping_mode", "")).strip()
            sentic_context = senticnet_vote(" ".join([str(lab.get("metadata", {}).get("raw_aspect", "")), str(lab.get("evidence_sentence", "")), review_text]).strip(), domain=domain)
            detail_rescued = _rescue_open_aspect_detail(lab, domain=domain, review_text=review_text)
            if detail_rescued:
                lab = dict(lab)
                lab["aspect"] = detail_rescued
                lab["implicit_aspect"] = detail_rescued
                lab["metadata"] = dict(lab.get("metadata", {}))
                lab["metadata"]["rescued_from_open_aspect"] = aspect
                lab["metadata"]["mapping_mode"] = "open_aspect_rescued_detail"
                lab["metadata"]["aspect_surface"] = detail_rescued
                lab["metadata"]["aspect_family"] = DETAIL_ALIAS_TO_FAMILY.get(detail_rescued, infer_aspect_family(detail_rescued, domain))
                lab["metadata"]["senticnet_concept"] = str(sentic_context.get("best_concept", ""))
                lab["metadata"]["senticnet_polarity"] = float(sentic_context.get("best_polarity", 0.0))
                lab["confidence"] = max(float(lab.get("confidence", 0.0)), 0.74)
                aspect = detail_rescued
                mode = "open_aspect_rescued_detail"
            rescued = _rescue_open_aspect(lab, domain=domain, review_text=review_text)
            if rescued:
                lab = dict(lab)
                lab["aspect"] = rescued
                lab["metadata"] = dict(lab.get("metadata", {}))
                lab["metadata"]["rescued_from_open_aspect"] = aspect
                lab["metadata"]["mapping_mode"] = "open_aspect_rescued"
                if not lab["metadata"].get("aspect_family"):
                    lab["metadata"]["aspect_family"] = infer_aspect_family(rescued, domain)
                lab["metadata"]["senticnet_concept"] = str(sentic_context.get("best_concept", ""))
                lab["metadata"]["senticnet_polarity"] = float(sentic_context.get("best_polarity", 0.0))
                lab["confidence"] = max(float(lab.get("confidence", 0.0)), 0.68)
                aspect = rescued
                mode = "open_aspect_rescued"

            # Prevent ontology errors: force unknown/vague buckets to 'unknown_aspect' or drop
            if aspect.startswith("other_") or aspect in {"quality", "experience", "generic"}:
                if mode != "open_aspect": # If it was a fallback, it's low quality
                     continue

            if mode == "open_aspect" and _should_preserve_open_aspect(aspect, domain=domain, freq=freq.get(aspect, 0), domain_agnostic_mode=domain_agnostic_mode):
                lab = dict(lab)
                lab["metadata"] = dict(lab.get("metadata", {}))
                lab["metadata"]["mapping_mode"] = "open_aspect_promoted"
                lab["confidence"] = max(float(lab.get("confidence", 0.0)), 0.62)
                mode = "open_aspect_promoted"

            if mode == "open_aspect" and freq.get(aspect, 0) < min_count:
                lab = dict(lab)
                lab["aspect"] = f"other_{domain}"
                lab["metadata"] = dict(lab.get("metadata", {}))
                lab["metadata"]["compacted_from"] = aspect
                lab["metadata"]["mapping_mode"] = "open_aspect_compacted"
                lab["confidence"] = max(float(lab.get("confidence", 0.0)), 0.5)
            key = (lab.get("aspect", ""), lab.get("sentiment", ""), lab.get("evidence_sentence", ""))
            if key not in merged or lab.get("confidence", 0) > merged[key].get("confidence", 0):
                merged[key] = lab
        row["labels"] = list(merged.values())
    return review_rows


def _tokenize_hint_text(*parts: str) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for token in re.findall(r"[a-z0-9_]+", normalize_text(part).lower()):
            if len(token) <= 2:
                continue
            tokens.add(token)
            if token.endswith("ies") and len(token) > 4:
                tokens.add(token[:-3] + "y")
            elif token.endswith("es") and len(token) > 4:
                tokens.add(token[:-2])
            elif token.endswith("s") and len(token) > 3:
                tokens.add(token[:-1])
    return tokens


def _rescue_open_aspect(label: Dict[str, Any], *, domain: str, review_text: str = "") -> str:
    metadata = label.get("metadata", {}) or {}
    mode = str(metadata.get("mapping_mode", "")).strip()
    if mode not in {"open_aspect", "open_aspect_compacted"}:
        return ""

    raw_aspect = str(metadata.get("raw_aspect") or label.get("aspect") or "").strip()
    evidence = str(label.get("evidence_sentence", "")).strip()
    sentiment = str(label.get("sentiment", "neutral")).strip().lower()
    tokens = _tokenize_hint_text(raw_aspect, evidence, review_text)
    if not tokens:
        return ""
    sentic_vote = senticnet_vote(" ".join([raw_aspect, evidence, review_text]).strip(), domain=domain)

    scored = []
    for canonical, hints in OPEN_ASPECT_RESCUE_HINTS.items():
        score = len(tokens.intersection(hints))
        score += float(sentic_vote.get("aspect_scores", {}).get(canonical, 0.0)) * 0.8
        if sentiment == "negative" and canonical in {"performance", "support_quality", "reliability", "delivery_logistics"}:
            score += 0.2
        if sentiment == "positive" and canonical in {"product_quality", "experience", "value", "usability"}:
            score += 0.2
        scored.append((score, canonical))
    scored.sort(reverse=True)
    best_score, best_aspect = scored[0]
    if best_score >= 2:
        return best_aspect

    if domain in DOMAIN_OPEN_FALLBACK and best_score >= 1:
        return best_aspect

    if domain in DOMAIN_OPEN_FALLBACK and raw_aspect and sentiment != "neutral":
        return DOMAIN_OPEN_FALLBACK[domain]
    return ""


def _rescue_open_aspect_detail(label: Dict[str, Any], *, domain: str, review_text: str = "") -> str:
    metadata = label.get("metadata", {}) or {}
    mode = str(metadata.get("mapping_mode", "")).strip()
    if mode not in {"open_aspect", "open_aspect_compacted", "open_aspect_rescued"}:
        return ""

    detail_hints = DOMAIN_DETAIL_HINTS.get(domain, {})
    if not detail_hints:
        return ""

    raw_aspect = str(metadata.get("raw_aspect") or label.get("aspect") or "").strip()
    evidence = str(label.get("evidence_sentence", "")).strip()
    tokens = _tokenize_hint_text(raw_aspect, evidence, review_text)
    if not tokens:
        return ""
    sentic_vote = senticnet_vote(" ".join([raw_aspect, evidence, review_text]).strip(), domain=domain)

    scored: List[tuple[float, str]] = []
    for detail_aspect, hints in detail_hints.items():
        hint_tokens = _tokenize_hint_text(detail_aspect, *hints)
        score = float(len(tokens.intersection(hint_tokens)))
        score += float(sentic_vote.get("aspect_scores", {}).get(detail_aspect, 0.0))
        if raw_aspect and normalize_text(raw_aspect).lower() == detail_aspect.replace("_", " "):
            score += 2.5
        if detail_aspect in tokens:
            score += 1.5
        scored.append((score, detail_aspect))

    scored.sort(reverse=True)
    best_score, best_detail = scored[0]
    return best_detail if best_score >= 2.0 else ""


def _should_preserve_open_aspect(aspect: str, *, domain: str, freq: int, domain_agnostic_mode: str) -> bool:
    if not aspect or aspect.startswith("other_"):
        return False
    if domain_agnostic_mode == "off":
        return False
    if domain_agnostic_mode != "always" and domain in CANONICAL_ASPECTS:
        return False
    token_count = len([p for p in aspect.split("_") if p])
    if token_count == 0 or token_count > 4:
        return False
    return freq >= 3


def _to_augmentation_base_record(row: Dict[str, Any]) -> Dict[str, Any] | None:
    aspects = []
    for label in row.get("labels", []):
        aspect = str(label.get("aspect", "")).strip()
        if not aspect or aspect.startswith("other_"):
            continue
        aspects.append(
            {
                "aspect_canonical": aspect,
                "sentiment": str(label.get("sentiment", "neutral")).strip().lower() or "neutral",
                "evidence_sentence": str(label.get("evidence_sentence", "")).strip(),
            }
        )
    if not aspects:
        return None
    return {
        "review_id": str(row.get("id", "")),
        "raw_text": str(row.get("review_text", "")),
        "clean_text": str(row.get("review_text", "")),
        "domain": str(row.get("domain", "generic")),
        "aspects": aspects,
    }


def _implicit_augmentation_budget(review_rows: List[Dict[str, Any]], target_ratio: float) -> int:
    explicit = sum(1 for row in review_rows for label in row.get("labels", []) if label.get("type") == "explicit")
    implicit = sum(1 for row in review_rows for label in row.get("labels", []) if label.get("type") == "implicit")
    if explicit + implicit == 0:
        return 0
    current_ratio = implicit / max(1, explicit + implicit)
    if current_ratio >= target_ratio:
        return 0
    needed = ((target_ratio * (explicit + implicit)) - implicit) / max(1e-6, 1.0 - target_ratio)
    return max(0, math.ceil(needed))


def _normalize_label_aliases(review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alias_map = {
        "batterylife": "battery_life",
        "techsupport": "support_quality",
        "warrenty": "support_quality",
        "extendedwarranty": "support_quality",
        "customer_service": "support_quality",
        "shipping": "delivery_logistics",
        "durability": "reliability",
    }
    for row in review_rows:
        for label in row.get("labels", []):
            aspect = str(label.get("aspect", "")).strip().lower()
            if aspect not in alias_map:
                continue
            target = alias_map[aspect]
            metadata = dict(label.get("metadata", {}))
            metadata["label_normalized_from"] = aspect
            metadata["aspect_family"] = infer_aspect_family(target, str(row.get("domain", "generic")))
            label["metadata"] = metadata
            label["aspect"] = target
    return review_rows


def _enrich_surface_details(review_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in review_rows:
        domain = str(row.get("domain", "generic")).strip().lower() or "generic"
        review_text = str(row.get("review_text", "")).strip()
        for label in row.get("labels", []):
            metadata = dict(label.get("metadata", {}))
            surface = normalize_text(metadata.get("aspect_surface") or label.get("implicit_aspect") or label.get("aspect") or "").lower().replace(" ", "_")
            aspect = str(label.get("aspect", "")).strip()
            if not aspect:
                continue
            detail_candidate = ""
            if aspect in BROAD_CANONICAL_ASPECTS or aspect.startswith("other_") or surface in SURFACE_STOPWORDS:
                detail_candidate = _rescue_open_aspect_detail(
                    {
                        **label,
                        "metadata": {
                            **metadata,
                            "mapping_mode": metadata.get("mapping_mode", "open_aspect"),
                            "raw_aspect": metadata.get("raw_aspect") or surface or aspect,
                        },
                    },
                    domain=domain,
                    review_text=review_text,
                )
                if not detail_candidate:
                    sentic = senticnet_vote(
                        " ".join(
                            [
                                str(metadata.get("raw_aspect", "")),
                                str(label.get("evidence_sentence", "")),
                                review_text,
                            ]
                        ).strip(),
                        domain=domain,
                    )
                    best_aspect = str(sentic.get("best_aspect", "")).strip()
                    if best_aspect and best_aspect not in BROAD_CANONICAL_ASPECTS and not best_aspect.startswith("other_"):
                        detail_candidate = best_aspect
                        metadata["senticnet_concept"] = str(sentic.get("best_concept", ""))
                        metadata["senticnet_polarity"] = float(sentic.get("best_polarity", 0.0))
            if detail_candidate:
                metadata["aspect_surface"] = detail_candidate
                label["implicit_aspect"] = detail_candidate
            label["metadata"] = metadata
    return review_rows


def _build_normalization_report(review_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = Counter()
    family_fallbacks = Counter()
    surface_to_canonical = Counter()
    broad_redistributions = Counter()
    for row in review_rows:
        for label in row.get("labels", []):
            metadata = dict(label.get("metadata", {}))
            if metadata.get("label_normalized_from"):
                normalized[f"{metadata['label_normalized_from']}->{label.get('aspect', '')}"] += 1
            if metadata.get("merged_from"):
                family_fallbacks[f"{metadata['merged_from']}->{label.get('aspect', '')}"] += 1
            if metadata.get("canonicalized_from"):
                surface_to_canonical[f"{metadata['canonicalized_from']}->{label.get('aspect', '')}"] += 1
            if metadata.get("redistributed_from"):
                broad_redistributions[f"{metadata['redistributed_from']}->{label.get('aspect', '')}"] += 1
    return {
        "normalized_aliases": dict(normalized),
        "family_fallbacks": dict(family_fallbacks),
        "surface_to_canonical": dict(surface_to_canonical),
        "broad_redistributions": dict(broad_redistributions),
    }


def _build_balanced_episodic_train(rows: List[Dict[str, Any]], max_share: float, *, min_surface_examples: int = 8, min_group_examples: int = 4) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    prepared: List[Dict[str, Any]] = []
    surface_counts = Counter()
    joint_counts = Counter()
    for row in rows:
        meta = dict(row.get("metadata", {}))
        surface = normalize_text(meta.get("aspect_surface") or row.get("implicit_aspect") or row.get("aspect") or "").lower().replace(" ", "_")
        family = normalize_text(meta.get("aspect_family") or row.get("aspect") or "").lower().replace(" ", "_")
        if surface:
            surface_counts[surface] += 1
        joint_counts[f"{row.get('aspect', '')}__{row.get('sentiment', 'neutral')}"] += 1

    for row in rows:
        cloned = {
            **row,
            "metadata": dict(row.get("metadata", {})),
        }
        aspect = str(cloned.get("aspect", "")).strip()
        sentiment = str(cloned.get("sentiment", "neutral")).strip().lower() or "neutral"
        meta = cloned["metadata"]
        surface = normalize_text(meta.get("aspect_surface") or cloned.get("implicit_aspect") or aspect).lower().replace(" ", "_")
        family = normalize_text(meta.get("aspect_family") or aspect).lower().replace(" ", "_")
        balanced_aspect = aspect
        if aspect.startswith("other_"):
            balanced_aspect = force_canonical_aspect(family or aspect, str(cloned.get("domain", "generic")))
            meta["balanced_reason"] = "canonical_other_fallback"
        elif joint_counts.get(f"{aspect}__{sentiment}", 0) < min_group_examples and family and family != aspect:
            balanced_aspect = family
            meta["balanced_reason"] = "family_grouping"
            meta["grouped_from"] = aspect
        elif aspect in BROAD_CANONICAL_ASPECTS and surface and surface_counts.get(surface, 0) >= min_surface_examples:
            detail_candidate = force_canonical_aspect(surface, str(cloned.get("domain", "generic")))
            if detail_candidate in EXPORT_CANONICAL_ASPECTS and detail_candidate != aspect:
                balanced_aspect = detail_candidate
                meta["balanced_reason"] = "canonical_detail"
                meta["grouped_from"] = aspect
        meta["original_aspect"] = aspect
        meta["balanced_aspect"] = balanced_aspect
        cloned["aspect"] = balanced_aspect
        prepared.append(cloned)

    by_aspect: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        by_aspect[str(row.get("aspect", ""))].append(row)
    total = len(prepared)
    cap = max(1, int(total * max_share))
    balanced: List[Dict[str, Any]] = []
    for aspect, aspect_rows in sorted(by_aspect.items()):
        ranked = sorted(
            aspect_rows,
            key=lambda item: (
                float(dict(item.get("metadata", {})).get("evidence_quality", 0.0)),
                1 if str(item.get("label_type", "explicit")) == "implicit" else 0,
            ),
            reverse=True,
        )
        kept = ranked[:cap] if len(ranked) > cap else ranked
        balanced.extend(kept)
    return balanced


def _text_signature(text: str) -> str:
    tokens = [t for t in normalize_text(text).lower().split() if len(t) > 2]
    return " ".join(tokens[:24])


def main() -> None:
    parser = argparse.ArgumentParser(description="ReviewOps dataset builder")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "input")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split-ratios", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--max-aspects", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--prefer-open-aspect", action="store_true")
    parser.add_argument("--domain-agnostic-mode", choices=["auto", "off", "always"], default="auto")
    parser.add_argument("--senticnet", dest="senticnet_enabled", action="store_true", default=True)
    parser.add_argument("--no-senticnet", dest="senticnet_enabled", action="store_false")
    parser.add_argument("--senticnet-resource-path", type=Path, default=PROJECT_ROOT / "resources" / "senticnet_seed.json")
    parser.add_argument("--min-implicit-vote-sources", type=int, default=2)
    parser.add_argument("--target-implicit-ratio", type=float, default=0.2)
    parser.add_argument("--episodic-max-aspect-share", type=float, default=0.35)
    parser.add_argument("--disable-second-aspect-extraction", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = [float(x.strip()) for x in args.split_ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--split-ratios must have 3 values that sum to 1.0")

    cfg = BuilderConfig(
        input_dir=_resolve_project_path(args.input_dir),
        output_dir=_resolve_project_path(args.output_dir),
        split_ratios={"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        random_seed=args.seed,
        max_aspects_per_review=args.max_aspects,
        confidence_threshold=args.confidence_threshold,
        prefer_canonical=not args.prefer_open_aspect,
        domain_agnostic_mode=args.domain_agnostic_mode,
        senticnet_enabled=args.senticnet_enabled,
        senticnet_resource_path=_resolve_project_path(args.senticnet_resource_path),
        min_implicit_vote_sources=args.min_implicit_vote_sources,
        target_implicit_ratio=args.target_implicit_ratio,
        episodic_max_aspect_share=args.episodic_max_aspect_share,
        conservative_second_aspect_extraction=not args.disable_second_aspect_extraction,
        dry_run=args.dry_run,
    )

    configure_senticnet(enabled=cfg.senticnet_enabled, resource_path=cfg.senticnet_resource_path)
    configure_inference(
        min_implicit_vote_sources=cfg.min_implicit_vote_sources,
        strong_senticnet_threshold=cfg.strong_senticnet_threshold,
        conservative_second_aspect_extraction=cfg.conservative_second_aspect_extraction,
    )

    files = list_input_files(cfg.input_dir)
    if not files:
        print(f"No supported input files found in: {cfg.input_dir}")
        return

    review_by_id: Dict[str, Dict[str, Any]] = {}
    seen_text_signatures: List[str] = []
    skipped: List[Dict[str, Any]] = []
    schema_reports: List[Dict[str, Any]] = []
    rows_read = 0

    for file_path in files:
        try:
            rows, file_type, columns = load_file_rows(file_path)
        except Exception as exc:
            skipped.append({"file": str(file_path), "row": None, "reason": f"file_load_error: {exc}"})
            continue

        rows_read += len(rows)
        schema = detect_schema(file_path, columns, file_type)
        if not schema.text_col:
            skipped.append({"file": str(file_path), "row": None, "reason": "missing_text_column"})
            continue
        sample_texts = [normalize_text(r.get(schema.text_col, "")) for r in rows[:150] if schema.text_col]
        sample_aspects: List[str] = []
        if schema.aspect_cols:
            for r in rows[:200]:
                sample_aspects.extend(extract_aspect_values(r, schema.aspect_cols))
        domain_guess = infer_domain(schema.file_path, schema.columns, sample_texts, sample_aspects)

        schema_reports.append(
            {
                "file": str(file_path),
                "file_type": file_type,
                "rows": len(rows),
                "schema": schema.__dict__,
                "inferred_domain": domain_guess,
            }
        )

        print(f"\nProcessing rows in {file_path.name}...")
        for idx, row in enumerate(tqdm(rows, desc="Labeling", leave=False)):
            review_text = normalize_text(row.get(schema.text_col, "")) if schema.text_col else ""
            if not review_text:
                skipped.append({"file": str(file_path), "row": idx, "reason": "empty_review_text"})
                continue
            sig = _text_signature(review_text)
            if any(sig == prev or len(set(sig.split()) & set(prev.split())) / max(1, len(set(sig.split()) | set(prev.split()))) >= 0.88 for prev in seen_text_signatures):
                skipped.append({"file": str(file_path), "row": idx, "reason": "near_duplicate_review_text"})
                continue
            seen_text_signatures.append(sig)

            source = file_path.name
            raw_id = normalize_text(row.get(schema.id_col, "")) if schema.id_col else ""
            if raw_id:
                row_id = f"{source}__{raw_id}"
            else:
                row_id = f"{source}__gen_{stable_hash(source, str(idx), review_text)}"

            split = normalize_text(row.get(schema.split_col, "")).lower() if schema.split_col else ""
            split = split if split in {"train", "val", "test"} else ""

            domain = normalize_text(row.get(schema.domain_col, "")).lower() if schema.domain_col else ""
            if not domain:
                domain = domain_guess

            aspect_values = extract_aspect_values(row, schema.aspect_cols)
            sentiment_value = row.get(schema.sentiment_col, "") if schema.sentiment_col else ""
            evidence_value = row.get(schema.evidence_col, "") if schema.evidence_col else ""
            span_from_value = row.get(schema.span_from_col, "") if schema.span_from_col else ""
            span_to_value = row.get(schema.span_to_col, "") if schema.span_to_col else ""

            labels = collect_labels_for_row(
                row=row,
                review_text=review_text,
                domain=domain,
                aspect_values=aspect_values,
                sentiment_value=sentiment_value,
                evidence_value=evidence_value,
                span_from_value=span_from_value,
                span_to_value=span_to_value,
                prefer_canonical=cfg.prefer_canonical,
                confidence_threshold=cfg.confidence_threshold,
            )

            if not labels:
                skipped.append({"file": str(file_path), "row": idx, "reason": "no_labels_after_inference"})
                continue

            if any(lab.get("type") == "implicit" and str(lab.get("aspect", "")).replace("_", " ") in normalize_text(review_text).lower() for lab in labels):
                labels = [lab for lab in labels if not (lab.get("type") == "implicit" and str(lab.get("aspect", "")).replace("_", " ") in normalize_text(review_text).lower())]
            if not labels:
                skipped.append({"file": str(file_path), "row": idx, "reason": "implicit_explicit_collision"})
                continue

            if len(labels) > cfg.max_aspects_per_review:
                labels = sorted(labels, key=lambda x: x.get("confidence", 0.0), reverse=True)[: cfg.max_aspects_per_review]

            entry = review_by_id.get(row_id)
            if entry is None:
                review_by_id[row_id] = {
                    "id": row_id,
                    "review_text": review_text,
                    "domain": domain,
                    "source": source,
                    "split": split,
                    "labels": labels,
                }
            else:
                if normalize_text(entry.get("review_text", "")) != review_text:
                    row_id = f"{row_id}__{stable_hash(review_text)}"
                    entry = review_by_id.get(row_id)
                if entry is None:
                    review_by_id[row_id] = {
                        "id": row_id,
                        "review_text": review_text,
                        "domain": domain,
                        "source": source,
                        "split": split,
                        "labels": labels,
                    }
                    continue
                combined = entry["labels"] + labels
                dedup = {}
                for lab in combined:
                    key = (lab["aspect"], lab.get("sentiment", ""), lab.get("evidence_sentence", ""))
                    if key not in dedup or lab.get("confidence", 0) > dedup[key].get("confidence", 0):
                        dedup[key] = lab
                entry["labels"] = list(dedup.values())[: cfg.max_aspects_per_review]
                if not entry.get("split"):
                    entry["split"] = split

    review_rows = list(review_by_id.values())
    
    # --- PHASE 2: Force Hybrid Branch Generation ---
    from implicit_augment import build_augmented_records
    # We use a limited LLM client if needed, or heuristic if not.
    # Note: BuilderConfig can be expanded for LLM API keys.
    augmented_branch = []
    known_domain_augment_budget = _implicit_augmentation_budget(review_rows, cfg.target_implicit_ratio)
    print("\nGenerating Hybrid Branch (Implicit Augmentation)...")
    for row in tqdm(review_rows, desc="Augmenting"):
        domain = str(row.get("domain", "generic")).strip().lower() or "generic"
        known_domain = domain in CANONICAL_ASPECTS
        if cfg.domain_agnostic_mode == "auto" and known_domain:
            if known_domain_augment_budget <= 0:
                continue
            if any(label.get("type") == "implicit" for label in row.get("labels", [])):
                continue
        base_record = _to_augmentation_base_record(row)
        if not base_record:
            continue
        augs = build_augmented_records(base_record, llm=None, implicit_query_only=True)
        row_aug_limit = 2 if known_domain_augment_budget > max(25, len(review_rows) // 12) else 1
        for a in augs[:row_aug_limit]:
            if len(a.get("preserved_aspects", [])) != len(a.get("preserved_sentiments", [])):
                continue
            if not normalize_text(a.get("clean_text", "")):
                continue
            labels = []
            for asp_can, sent in zip(a["preserved_aspects"], a["preserved_sentiments"]):
                final_aspect = force_canonical_aspect(asp_can, domain)
                evidence_sentence = _shorten_augmentation_evidence(str(a["clean_text"]), final_aspect or asp_can)
                evidence_meta = {
                    **build_evidence_metadata(str(a["clean_text"]), evidence_sentence, final_aspect or asp_can, raw_aspect=asp_can),
                    "augmentation_sentence_evidence": True,
                }
                labels.append({
                    "aspect": final_aspect or asp_can,
                    "implicit_aspect": asp_can,
                    "sentiment": sent,
                    "evidence_sentence": evidence_sentence,
                    "type": "implicit",
                    "confidence": 0.84,
                    "metadata": {
                        "rule": "augmentation",
                        "orig_id": a["source_record_id"],
                        "augmentation_type": a.get("augmentation_type", ""),
                        "aspect_surface": asp_can,
                        "aspect_family": infer_aspect_family(final_aspect or asp_can, domain),
                        "vote_sources": ["augmentation"],
                        **evidence_meta,
                    }
                })
            if not labels:
                continue
            augmented_branch.append(
                {
                    "id": str(a["review_id"]),
                    "review_text": str(a["clean_text"]),
                    "domain": str(row.get("domain", "generic")),
                    "source": f"{row.get('source', 'unknown')}::augmented",
                    "split": str(row.get("split", "")),
                    "labels": labels,
                }
            )
            if cfg.domain_agnostic_mode == "auto" and known_domain:
                known_domain_augment_budget -= len(labels)
                if known_domain_augment_budget <= 0:
                    break

    review_rows.extend(augmented_branch)
    # --- END PHASE 2 ---

    review_rows = compact_open_aspects(review_rows, domain_agnostic_mode=cfg.domain_agnostic_mode)
    review_rows = _enrich_surface_details(review_rows)
    review_rows = _normalize_label_aliases(review_rows)
    review_rows = _canonicalize_export_labels(review_rows)
    non_empty_rows = []
    for row in review_rows:
        if row.get("labels"):
            non_empty_rows.append(row)
        else:
            skipped.append({"file": str(row.get("source", "")), "row": row.get("id"), "reason": "no_labels_after_compaction"})
    review_rows = non_empty_rows

    # --- PHASE 3: ProtoNet Class Merger (Stability) ---
    # Collapse rare aspects (< 5 examples) into their parent universal superclasses
    # Pre-build reverse lookup: alias -> superclass
    alias_to_super = {}
    for domain_cat in CANONICAL_ASPECTS.values():
        for supercl, aliases in domain_cat.items():
            for a in aliases:
                alias_to_super[a] = supercl
                
    counts = Counter(l.get("aspect") for r in review_rows for l in r.get("labels", []))
    for row in review_rows:
        for lab in row.get("labels", []):
            aspect = str(lab.get("aspect", "")).strip()
            if not aspect:
                continue
            if counts[aspect] < 5:
                if aspect in alias_to_super:
                    lab["metadata"] = dict(lab.get("metadata", {}))
                    lab["metadata"]["merged_from"] = aspect
                    lab["aspect"] = force_canonical_aspect(alias_to_super[aspect], str(row.get("domain", "generic")))
                elif "_" in aspect and not aspect.startswith("other_"):
                    # heuristic like 'battery_life' -> 'performance'
                    parts = aspect.split("_")
                    if parts[0] in {"battery", "screen", "keyboard", "waiter", "service"}:
                         # these are high-probability performance or service sub-labels
                         lab["metadata"] = dict(lab.get("metadata", {}))
                         lab["metadata"]["merged_from"] = aspect
                         lab["aspect"] = "performance" if parts[0] != "waiter" else "support_quality"
    # --- END PHASE 3 ---
    review_rows = _canonicalize_export_labels(review_rows)

    # --- PHASE 4: Demo Dominance Control (Final Pruning) ---
    # 1. Cap 'Other' aspects to 15% of total row count
    other_threshold = int(len(review_rows) * 0.15)
    final_rows = []
    other_counts = defaultdict(int)
    for row in review_rows:
        aspects = [l.get("aspect", "") for l in row.get("labels", [])]
        if not aspects:
            skipped.append({"file": str(row.get("source", "")), "row": row.get("id"), "reason": "no_labels_before_final_prune"})
            continue
        is_only_other = all(a.startswith("other_") for a in aspects)
        if is_only_other:
            primary_other = aspects[0]
            if other_counts[primary_other] < other_threshold:
                 other_counts[primary_other] += 1
                 final_rows.append(row)
            # drop excessive other-only rows
        else:
            # Keep all rows that have at least one meaningful (non-other) aspect
            final_rows.append(row)
    
    # 1.5 Final cap per label (Fix for class imbalance)
    MAX_SAMPLES_PER_LABEL = 500
    label_counts = Counter()
    import random
    rng = random.Random(cfg.random_seed)
    rng.shuffle(final_rows)
    for row in final_rows:
        row["labels"] = sorted(row.get("labels", []), key=lambda x: x.get("confidence", 0.0), reverse=True)
        new_labels = []
        for l in row.get("labels", []):
            aspect = l.get("aspect")
            if label_counts[aspect] < MAX_SAMPLES_PER_LABEL:
                new_labels.append(l)
                label_counts[aspect] += 1
        row["labels"] = new_labels

    # 2. Final ultra-rare purge (< 5 samples)
    curr_counts = Counter(l.get("aspect") for r in final_rows for l in r.get("labels", []))
    for row in final_rows:
        row["labels"] = [l for l in row.get("labels", []) if curr_counts[l.get("aspect")] >= 5]
    
    # 3. Final drop of rows with no labels after purge
    review_rows = [r for r in final_rows if r.get("labels")]
    # --- END PHASE 4 ---
    review_rows = _canonicalize_export_labels(review_rows)
    for row in review_rows:
        row["labels"] = [label for label in row.get("labels", []) if str(label.get("aspect", "")).strip() in EXPORT_CANONICAL_ASPECTS]
    review_rows = [row for row in review_rows if row.get("labels")]

    review_rows = assign_splits(review_rows, cfg.split_ratios, seed=cfg.random_seed)
    review_rows = apply_domain_mixing(review_rows, max_open_share=cfg.open_corpora_max_share, gold_eval_only=cfg.gold_benchmark_eval_only)


    def format_target_text(labels: List[Dict]) -> str:
        explicit = []
        implicit = []
        for lab in labels:
            aspect = str(lab.get("aspect", "")).strip()
            sentiment = str(lab.get("sentiment", "neutral")).strip()
            span = str(lab.get("evidence_sentence", "")).strip() or "[no_span]"
            token = f"{aspect} | {sentiment} | {span}"
            if lab.get("type", "explicit") == "explicit":
                explicit.append((aspect, token))
            else:
                implicit.append((aspect, token))
        explicit.sort(key=lambda x: x[0])
        implicit.sort(key=lambda x: x[0])
        return " ;; ".join([t[1] for t in explicit] + [t[1] for t in implicit])

    for row in review_rows:
        row["target_text"] = format_target_text(row.get("labels", []))

    split_review = split_rows(review_rows)


    all_episode_rows = build_episodic_rows(review_rows)
    # Enforce >= 10 sample minimum for episodic classes
    episode_counts = Counter(f"{r.get('aspect', '')}::{r.get('sentiment', '')}" for r in all_episode_rows)
    all_episode_rows = [r for r in all_episode_rows if episode_counts[f"{r.get('aspect', '')}::{r.get('sentiment', '')}"] >= 10]
    
    split_episode = {k: [r for r in all_episode_rows if r.get("split") == k] for k in ["train", "val", "test"]}
    balanced_episode_train = _build_balanced_episodic_train(split_episode["train"], cfg.episodic_max_aspect_share)
    normalization_report = _build_normalization_report(review_rows)

    summary = {
        "datasets_processed": len(schema_reports),
        "total_rows_read": rows_read,
        "schema_reports": schema_reports,
        "reviewlevel_rows": len(review_rows),
        "episodic_rows": len(all_episode_rows),
        "explicit_labels": sum(1 for r in review_rows for l in r.get("labels", []) if l.get("type") == "explicit"),
        "implicit_labels": sum(1 for r in review_rows for l in r.get("labels", []) if l.get("type") == "implicit"),
        "unique_aspects": sorted(list({l.get("aspect") for r in review_rows for l in r.get("labels", []) if l.get("aspect")})),
        "top_aspects": summarize_aspects(review_rows),
        "split_sizes": {
            "reviewlevel": {k: len(v) for k, v in split_review.items()},
            "episodic": {k: len(v) for k, v in split_episode.items()},
        },
        "duplicate_id_count": validate_review_rows(review_rows).get("duplicate_id", 0),
        "rows_skipped": len(skipped),
        "rows_skipped_reasons": dict(Counter(s["reason"] for s in skipped)),
        "validation": {
            "reviewlevel": validate_review_rows(review_rows),
            "leakage_ids": leakage_ids(split_review),
            "few_shot_warnings": few_shot_warnings(all_episode_rows),
            "aspect_frequency": aspect_frequency(review_rows),
        },
        "senticnet": senticnet_status(),
        "balanced_episodic_train_rows": len(balanced_episode_train),
    }

    data_quality_report = build_data_quality_report(review_rows, all_episode_rows)
    data_quality_report["senticnet"] = senticnet_status()
    data_quality_report["balanced_episodic_train_rows"] = len(balanced_episode_train)
    label_issue_candidates = build_label_issue_candidates(review_rows)
    episode_readiness_report = build_episode_readiness_report(
        all_episode_rows,
        n_way=cfg.n_way,
        k_shot=cfg.k_shot,
        q_query=cfg.q_query,
    )
    episode_readiness_report["balanced_train"] = build_episode_readiness_report(
        balanced_episode_train,
        n_way=cfg.n_way,
        k_shot=cfg.k_shot,
        q_query=cfg.q_query,
    )["splits"].get("train", {})

    print("=== Dataset Builder Summary ===")
    print(f"Datasets processed: {summary['datasets_processed']}")
    print(f"Rows read: {summary['total_rows_read']}")
    print(f"Review-level rows: {summary['reviewlevel_rows']}")
    print(f"Episodic rows: {summary['episodic_rows']}")
    print(f"Split sizes: {summary['split_sizes']}")
    print(f"Skipped rows: {summary['rows_skipped']} -> {summary['rows_skipped_reasons']}")

    preview_reviews = review_rows[: cfg.sample_preview_count]
    preview_episodic = all_episode_rows[: cfg.sample_preview_count]
    print("\nReview-level preview:")
    for row in preview_reviews:
        print({"id": row["id"], "domain": row["domain"], "split": row["split"], "labels": row["labels"][:2]})

    print("\nEpisodic preview:")
    for row in preview_episodic:
        print({k: row[k] for k in ["example_id", "parent_review_id", "aspect", "implicit_aspect", "split"]})

    if cfg.dry_run:
        print("\nDry run mode: no files written.")
        return

    ensure_output_dirs(cfg.output_dir)
    out_review = cfg.output_dir / "reviewlevel"
    out_epi = cfg.output_dir / "episodic"
    out_reports = cfg.output_dir / "reports"

    for split in ["train", "val", "test"]:
        write_jsonl(out_review / f"{split}.jsonl", split_review[split])
        write_jsonl(out_epi / f"{split}.jsonl", split_episode[split])
    write_jsonl(out_epi / "train_balanced.jsonl", balanced_episode_train)

    write_json(out_reports / "build_report.json", summary)
    write_json(out_reports / "data_quality_report.json", data_quality_report)
    write_json(out_reports / "episode_readiness_report.json", episode_readiness_report)
    write_json(out_reports / "normalization_report.json", normalization_report)
    write_jsonl(out_reports / "label_issue_candidates.jsonl", label_issue_candidates)
    write_jsonl(out_reports / "skipped_rows.jsonl", skipped)

    jsonl_errors = []
    for split in ["train", "val", "test"]:
        jsonl_errors.extend(validate_jsonl(out_review / f"{split}.jsonl"))
        jsonl_errors.extend(validate_jsonl(out_epi / f"{split}.jsonl"))
    if jsonl_errors:
        print("\nValidation JSONL errors found:")
        for err in jsonl_errors[:15]:
            print(" -", err)
    else:
        print("\nJSONL validation passed.")


if __name__ == "__main__":
    main()
