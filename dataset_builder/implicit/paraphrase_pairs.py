from __future__ import annotations


def generate_explicit_implicit_pairs(explicit_aspect: str, symptom: str) -> tuple[str, str]:
    return (str(explicit_aspect).strip(), str(symptom).strip())


def load_pair_bank() -> list[tuple[str, str]]:
    return []
