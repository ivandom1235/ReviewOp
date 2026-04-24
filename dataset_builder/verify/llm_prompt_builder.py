from __future__ import annotations
from typing import Any

def build_verifier_prompt(review_text: str, candidates: list[dict[str, Any]]) -> str:
    """Build a detailed prompt for aspect-level verification."""
    candidate_list = ""
    for i, c in enumerate(candidates):
        candidate_list += f"Index {i}: {c['aspect_raw']} | {c['sentiment']} | Evidence: '{c['evidence_text']}'\n"
        
    return (
        "You are an expert data auditor for Aspect-Based Sentiment Analysis (ABSA).\n"
        "Your task is to verify a list of candidate interpretations extracted from a review.\n\n"
        f"Review: {review_text}\n\n"
        "Candidates:\n"
        f"{candidate_list}\n"
        "Instructions:\n"
        "For each candidate, decide on one of the following actions:\n"
        "- 'keep': The aspect and sentiment are correctly grounded in the evidence.\n"
        "- 'drop': The aspect is noisy (e.g., 'The', 'World'), irrelevant, or the sentiment/evidence is wrong.\n"
        "- 'merge': This candidate is a duplicate of another candidate (indicate which index to merge into).\n"
        "- 'add': Suggest a missing aspect that you noticed in the text but is not in the list.\n\n"
        "Return the results as a VALID JSON list of objects, one per candidate, with fields: 'index', 'action', 'reason'.\n"
        "IMPORTANT: Ensure the output is a single valid JSON list with commas correctly placed between objects.\n"
        "Ensure all double quotes inside the 'reason' string are properly escaped with a backslash (e.g., \\\"quote\\\").\n"
        "Do not include any conversational filler; return ONLY the raw JSON list.\n"
        "Example: [{\"index\": 0, \"action\": \"keep\", \"reason\": \"correctly identifies aspect and sentiment\"}, {\"index\": 1, \"action\": \"drop\", \"reason\": \"noisy aspect\"}]"
    )
