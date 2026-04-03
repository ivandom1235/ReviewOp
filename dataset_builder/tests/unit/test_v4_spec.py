from __future__ import annotations

from pathlib import Path
import unittest


class V41SpecTests(unittest.TestCase):
    def test_v41_spec_contains_required_novelty_sections(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        spec_path = repo_root / "V4.docx"
        content = spec_path.read_text(encoding="utf-8")
        required = [
            "V4.1 Domain-Agnostic and Novelty-Preserved Specification",
            "Method Identity",
            "Novelty Guardrail",
            "Required ablations",
            "Comparative Research Matrix",
        ]
        for marker in required:
            self.assertIn(marker, content)


if __name__ == "__main__":
    unittest.main()
