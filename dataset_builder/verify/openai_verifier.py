from __future__ import annotations
import json
import os
from dataclasses import asdict
from typing import Any

from ..schemas.interpretation import Interpretation
from ..llm.provider_factory import get_llm_client
from .llm_prompt_builder import build_verifier_prompt
from .llm_response_parser import parse_keep_drop_merge_add, validate_verifier_json

class OpenAIVerifier:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.client = get_llm_client(cfg)

    def verify(self, review_text: str, interpretations: list[Interpretation]) -> list[dict[str, Any]]:
        """Verify multiple interpretations for a single review."""
        if not interpretations:
            return []
            
        candidates = [asdict(i) for i in interpretations]
        prompt = build_verifier_prompt(review_text, candidates)
        
        try:
            response_text = self.client.generate(prompt, max_tokens=2048).strip()
            # Find the JSON block in the response (robustly)
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                try:
                    decisions = parse_keep_drop_merge_add(json_str)
                except json.JSONDecodeError:
                    # Fallback: attempt to fix common errors like trailing commas or missing commas between objects
                    import re
                    cleaned_json = re.sub(r'\}\s*\{', '}, {', json_str)
                    cleaned_json = cleaned_json.strip().replace("\n", " ")
                    decisions = parse_keep_drop_merge_add(cleaned_json)

                if validate_verifier_json(decisions):
                    return decisions
        except Exception as e:
            print(f"Verifier failed: {e}")
            
        # Default: keep all if LLM fails
        return [{"action": "keep", "index": i} for i in range(len(interpretations))]

    def verify_row(self, row: Any) -> Any:
        """Helper to verify a whole row and return the updated row."""
        from dataclasses import replace
        decisions = self.verify(row.review_text, list(row.gold_interpretations))
        
        new_gold = []
        orig = list(row.gold_interpretations)
        
        for dec in decisions:
            action = dec.get("action", "keep")
            idx = dec.get("index")
            
            if action == "keep" and idx is not None and 0 <= idx < len(orig):
                new_gold.append(orig[idx])
            elif action == "add":
                aspect_raw = dec.get("aspect", "unknown")
                # Populate anchor fields for later canonicalization
                new_gold.append(Interpretation(
                    aspect_raw=aspect_raw,
                    aspect_canonical="unknown",
                    latent_family="unknown",
                    label_type="implicit",
                    sentiment="unknown",
                    evidence_text=dec.get("evidence", row.review_text),
                    evidence_span=[0, len(row.review_text)],
                    source="llm_verifier",
                    support_type="contextual",
                    source_type="implicit_llm",
                    aspect_anchor=aspect_raw.split()[-1].lower() if aspect_raw != "unknown" else None,
                    anchor_source="llm_added"
                ))
            # drop and merge are handled by omission or specific logic
            # for now we keep it simple to avoid metadata loss
                
        return replace(row, gold_interpretations=tuple(new_gold))
