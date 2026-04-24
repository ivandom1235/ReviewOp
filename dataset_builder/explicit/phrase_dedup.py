from __future__ import annotations


def dedup_phrases_exact(phrases: list[str]) -> list[str]:
    return list(dict.fromkeys(phrases))


def dedup_phrases_embedding(phrases: list[str]) -> list[str]:
    return dedup_phrases_exact(phrases)
