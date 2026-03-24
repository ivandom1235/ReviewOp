from __future__ import annotations

from typing import Any, Dict, List

from progress import track


def _normalized_label_type(raw: Any) -> str:
    label_type = str(raw or "explicit").strip().lower()
    return label_type or "explicit"


def adapt_reviewlevel_rows(rows_by_split: Dict[str, List[Dict[str, Any]]], *, progress_enabled: bool) -> Dict[str, List[Dict[str, Any]]]:
    adapted: Dict[str, List[Dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        out: List[Dict[str, Any]] = []
        for row in track(rows, total=len(rows), desc=f"adapt:{split}", enabled=progress_enabled):
            review_id = str(row.get("id") or row.get("review_id") or "").strip()
            review_text = str(row.get("clean_text") or row.get("review_text") or "").strip()
            domain = str(row.get("domain") or "unknown").strip().lower() or "unknown"
            source = str(row.get("source") or row.get("source_file") or "reviewlevel").strip()
            labels = sorted(
                list(row.get("labels", [])),
                key=lambda item: (
                    str(item.get("aspect", "")),
                    str(item.get("sentiment", "")),
                    str(item.get("evidence_sentence", "")),
                ),
            )
            for index, label in enumerate(labels, start=1):
                aspect = str(label.get("aspect") or label.get("implicit_aspect") or "").strip()
                if not aspect:
                    continue
                evidence = str(label.get("evidence_sentence") or review_text).strip() or review_text
                sentiment = str(label.get("sentiment") or "neutral").strip().lower() or "neutral"
                out.append(
                    {
                        "example_id": f"{review_id}_e{index}",
                        "parent_review_id": review_id,
                        "review_text": review_text,
                        "evidence_sentence": evidence,
                        "domain": domain,
                        "aspect": aspect,
                        "implicit_aspect": aspect,
                        "sentiment": sentiment,
                        "label_type": _normalized_label_type(label.get("type")),
                        "split": split,
                        "source": source,
                        "confidence": float(label.get("confidence", 1.0)),
                    }
                )
        adapted[split] = out
    return adapted
