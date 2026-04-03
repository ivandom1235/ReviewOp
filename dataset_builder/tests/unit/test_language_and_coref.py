from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from coref import heuristic_coref
from language_utils import detect_language, is_implicit_ready


class LanguageAndCorefTests(unittest.TestCase):
    def test_detect_language_and_implicit_readiness(self) -> None:
        self.assertEqual(detect_language("The battery life is great"), "en")
        self.assertEqual(detect_language("La pantalla es brillante"), "es")
        self.assertEqual(detect_language("Le service est tres bon"), "fr")
        self.assertEqual(detect_language("La service y pantalla"), "mixed")
        self.assertTrue(is_implicit_ready("The battery life is great and the screen is bright", language="en", min_tokens=8, supported_languages=("en", "es")))
        self.assertFalse(is_implicit_ready("Great", language="en", min_tokens=8, supported_languages=("en", "es")))

    def test_heuristic_coref_rewrites_pronouns(self) -> None:
        result = heuristic_coref("The laptop is fast. It is also light.")
        self.assertIn("laptop", result.text.lower())
        self.assertGreaterEqual(len(result.chains), 1)


if __name__ == "__main__":
    unittest.main()
