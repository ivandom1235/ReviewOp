ASPECT_MAP_PROMPT = """You are an ABSA ontology assistant.
Map each aspect phrase to a concise canonical snake_case label.
Keep semantic intent and avoid over-specific labels.
Return strict JSON: {\"mappings\": [{\"raw\": ..., \"canonical\": ..., \"confidence\": 0-1}]}
"""

IMPLICIT_REWRITE_PROMPT = """Rewrite the review to express the same aspect sentiment more implicitly.
Constraints:
- preserve original sentiment per aspect
- keep realistic user language
- do not add new aspects
Return strict JSON: {\"rewrites\": [{\"text\": ..., \"augmentation_type\": ...}]}
"""
