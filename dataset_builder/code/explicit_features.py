from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import normalize_whitespace, token_count


@dataclass
class ExplicitArtifacts:
    numeric_scaler: MinMaxScaler | None
    categorical_encoder: OneHotEncoder | None
    train_categories: Dict[str, List[str]]
    label_maps: Dict[str, Dict[str, int]]


def text_stats(text: str) -> Dict[str, Any]:
    clean = normalize_whitespace(text)
    words = clean.split()
    chars = len(clean)
    letters = sum(1 for ch in clean if ch.isalpha())
    digits = sum(1 for ch in clean if ch.isdigit())
    punctuation = sum(1 for ch in clean if not ch.isalnum() and not ch.isspace())
    return {
        "char_count": chars,
        "word_count": len(words),
        "sentence_count": max(1, clean.count(".") + clean.count("!") + clean.count("?")),
        "avg_word_length": round(sum(len(word) for word in words) / max(1, len(words)), 4),
        "unique_word_ratio": round(len(set(word.lower() for word in words)) / max(1, len(words)), 4),
        "punctuation_ratio": round(punctuation / max(1, chars), 4),
        "capitalisation_ratio": round(sum(1 for ch in clean if ch.isupper()) / max(1, letters), 4),
        "exclamation_count": clean.count("!"),
        "question_count": clean.count("?"),
        "digit_ratio": round(digits / max(1, chars), 4),
        "token_count": token_count(text),
    }


def fit_explicit_artifacts(train_frame: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> ExplicitArtifacts:
    numeric_scaler = MinMaxScaler() if numeric_columns else None
    if numeric_scaler is not None:
        numeric_values = train_frame[numeric_columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        numeric_scaler.fit(numeric_values)

    categorical_encoder = None
    if categorical_columns:
        try:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        categorical_encoder.fit(train_frame[categorical_columns].fillna("unknown").astype(str))

    categories = {col: sorted(train_frame[col].fillna("unknown").astype(str).unique().tolist()) for col in categorical_columns}
    label_maps = {col: {category: idx for idx, category in enumerate(categories[col])} for col in categorical_columns}
    return ExplicitArtifacts(numeric_scaler=numeric_scaler, categorical_encoder=categorical_encoder, train_categories=categories, label_maps=label_maps)


def build_explicit_row(
    row: Dict[str, Any],
    *,
    artifacts: ExplicitArtifacts,
    numeric_columns: List[str],
    categorical_columns: List[str],
    datetime_columns: List[str],
    text_column: str | None,
) -> Dict[str, Any]:
    explicit: Dict[str, Any] = {}
    if numeric_columns:
        numeric_values = pd.DataFrame([row]).reindex(columns=numeric_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = artifacts.numeric_scaler.transform(numeric_values) if artifacts.numeric_scaler is not None else numeric_values.values
        for idx, column in enumerate(numeric_columns):
            explicit[column] = float(numeric_values.iloc[0][column])
            explicit[f"{column}_norm"] = float(scaled[0][idx])
    for column in categorical_columns:
        value = str(row.get(column, "unknown")).strip() or "unknown"
        explicit[column] = value
        categories = artifacts.train_categories.get(column, [])
        label_map = artifacts.label_maps.get(column, {})
        explicit[f"{column}_id"] = int(label_map.get(value, -1))
        if len(categories) <= 50:
            for category in categories:
                explicit[f"{column}_{category}"] = int(value == category)
    for column in datetime_columns:
        value = row.get(column)
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.notna(parsed):
            explicit[f"{column}_year"] = int(parsed.year)
            explicit[f"{column}_month"] = int(parsed.month)
            explicit[f"{column}_day_of_week"] = int(parsed.dayofweek)
            explicit[f"{column}_is_weekend"] = int(parsed.dayofweek >= 5)
    if text_column:
        explicit[f"{text_column}_stats"] = text_stats(str(row.get(text_column, "")).strip())
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_file": row.get("source_file"),
        "source_text": str(row.get(text_column, "")).strip() if text_column else "",
        "explicit": explicit,
    }
