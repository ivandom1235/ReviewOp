from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils import normalize_whitespace, token_count


@dataclass
class NumericFeaturiser:
    columns: List[str]
    scaler: MinMaxScaler | None = None

    def fit(self, frame: pd.DataFrame) -> "NumericFeaturiser":
        if self.columns:
            numeric_values = frame[self.columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            self.scaler = MinMaxScaler()
            self.scaler.fit(numeric_values)
        return self

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        explicit: Dict[str, Any] = {}
        if not self.columns:
            return explicit
        numeric_values = pd.DataFrame([row]).reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scaled = self.scaler.transform(numeric_values) if self.scaler is not None else numeric_values.values
        for idx, column in enumerate(self.columns):
            explicit[column] = float(numeric_values.iloc[0][column])
            explicit[f"{column}_norm"] = float(scaled[0][idx])
        return explicit


@dataclass
class CategoricalFeaturiser:
    columns: List[str]
    encoder: OneHotEncoder | None = None
    categories: Dict[str, List[str]] | None = None
    label_maps: Dict[str, Dict[str, int]] | None = None

    def fit(self, frame: pd.DataFrame) -> "CategoricalFeaturiser":
        if not self.columns:
            self.categories = {}
            self.label_maps = {}
            return self
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.encoder.fit(frame[self.columns].fillna("unknown").astype(str))
        self.categories = {col: sorted(frame[col].fillna("unknown").astype(str).unique().tolist()) for col in self.columns}
        self.label_maps = {col: {category: idx for idx, category in enumerate(categories)} for col, categories in self.categories.items()}
        return self

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        explicit: Dict[str, Any] = {}
        for column in self.columns:
            value = str(row.get(column, "unknown")).strip() or "unknown"
            explicit[column] = value
            categories = (self.categories or {}).get(column, [])
            label_map = (self.label_maps or {}).get(column, {})
            explicit[f"{column}_id"] = int(label_map.get(value, -1))
            if len(categories) <= 50:
                for category in categories:
                    explicit[f"{column}_{category}"] = int(value == category)
        return explicit


@dataclass
class DatetimeFeaturiser:
    columns: List[str]

    def fit(self, frame: pd.DataFrame) -> "DatetimeFeaturiser":
        return self

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        explicit: Dict[str, Any] = {}
        for column in self.columns:
            parsed = pd.to_datetime(row.get(column), errors="coerce", utc=True)
            if pd.notna(parsed):
                explicit[f"{column}_year"] = int(parsed.year)
                explicit[f"{column}_month"] = int(parsed.month)
                explicit[f"{column}_day_of_week"] = int(parsed.dayofweek)
                explicit[f"{column}_is_weekend"] = int(parsed.dayofweek >= 5)
        return explicit


@dataclass
class TextStatsFeaturiser:
    column: str | None

    def fit(self, frame: pd.DataFrame) -> "TextStatsFeaturiser":
        return self

    def transform(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not self.column:
            return {}
        text = normalize_whitespace(str(row.get(self.column, "")).strip())
        words = text.split()
        chars = len(text)
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        punctuation = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        return {
            f"{self.column}_stats": {
                "char_count": chars,
                "word_count": len(words),
                "sentence_count": max(1, text.count(".") + text.count("!") + text.count("?")),
                "avg_word_length": round(sum(len(word) for word in words) / max(1, len(words)), 4),
                "unique_word_ratio": round(len(set(word.lower() for word in words)) / max(1, len(words)), 4),
                "punctuation_ratio": round(punctuation / max(1, chars), 4),
                "capitalisation_ratio": round(sum(1 for ch in text if ch.isupper()) / max(1, letters), 4),
                "exclamation_count": text.count("!"),
                "question_count": text.count("?"),
                "digit_ratio": round(digits / max(1, chars), 4),
                "token_count": token_count(text),
            }
        }


@dataclass
class ExplicitArtifacts:
    numeric: NumericFeaturiser
    categorical: CategoricalFeaturiser
    datetime: DatetimeFeaturiser
    text: TextStatsFeaturiser


def fit_explicit_artifacts(train_frame: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> ExplicitArtifacts:
    text_column = None
    for candidate in train_frame.columns:
        if candidate not in set(numeric_columns) | set(categorical_columns) | {"source_file", "split", "id"}:
            if train_frame[candidate].astype(str).map(token_count).mean() >= 3:
                text_column = candidate
                break
    numeric = NumericFeaturiser(numeric_columns).fit(train_frame)
    categorical = CategoricalFeaturiser(categorical_columns).fit(train_frame)
    datetime_columns = [column for column in train_frame.columns if column.endswith("_at") or column.endswith("_date")]
    datetime = DatetimeFeaturiser(datetime_columns).fit(train_frame)
    text = TextStatsFeaturiser(text_column).fit(train_frame)
    return ExplicitArtifacts(numeric=numeric, categorical=categorical, datetime=datetime, text=text)


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
    explicit.update(artifacts.numeric.transform(row))
    explicit.update(artifacts.categorical.transform(row))
    explicit.update(artifacts.datetime.transform(row))
    if text_column:
        explicit.update(TextStatsFeaturiser(text_column).transform(row))
    for column in datetime_columns:
        parsed = pd.to_datetime(row.get(column), errors="coerce", utc=True)
        if pd.notna(parsed):
            explicit[f"{column}_year"] = int(parsed.year)
            explicit[f"{column}_month"] = int(parsed.month)
            explicit[f"{column}_day_of_week"] = int(parsed.dayofweek)
            explicit[f"{column}_is_weekend"] = int(parsed.dayofweek >= 5)
    return {
        "id": row.get("id"),
        "split": row.get("split"),
        "source_file": row.get("source_file"),
        "source_text": str(row.get(text_column, "")).strip() if text_column else "",
        "explicit": explicit,
    }
