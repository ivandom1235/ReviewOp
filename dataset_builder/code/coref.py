from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorefResult:
    text: str
    chains: list[dict]


def no_coref(text: str) -> CorefResult:
    return CorefResult(text=text, chains=[])


def heuristic_coref(text: str) -> CorefResult:
    return CorefResult(text=text, chains=[])
