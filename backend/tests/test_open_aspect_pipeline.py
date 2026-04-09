from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
for extra_path in (ROOT, ROOT / "backend"):
    extra_str = str(extra_path)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from services.open_aspect import extract_open_aspects, open_aspect_model_status


class OpenAspectPipelineTests(unittest.TestCase):
    SAMPLE_TEXT = (
        "I’ve been using this smartwatch for about three weeks now. "
        "The battery life easily lasts two full days, which is much better than my previous watch. "
        "However, the heart rate sensor is inconsistent during workouts, and sometimes it spikes randomly. "
        "The display is bright and readable even in sunlight, but the strap feels cheap and uncomfortable after long hours. "
        "The app integration is smooth, though syncing occasionally takes longer than expected"
    )

    def test_model_status_reports_installation_state(self) -> None:
        status = open_aspect_model_status()
        self.assertIn("available", status)
        self.assertEqual(status["model"], "en_core_web_sm")

    def test_extract_open_aspects_returns_specific_aspects_when_model_is_available(self) -> None:
        status = open_aspect_model_status()
        if not status.get("available"):
            self.skipTest(status.get("error") or "spaCy model is unavailable")

        aspects = extract_open_aspects(self.SAMPLE_TEXT, max_aspects=8)
        self.assertTrue(aspects)
        self.assertNotEqual(aspects, ["general"])
        self.assertTrue(any("battery" in aspect or "display" in aspect or "strap" in aspect for aspect in aspects))

    def test_review_pipeline_surfaces_missing_spacy_model(self) -> None:
        with patch("services.open_aspect.spacy.load", side_effect=OSError("missing model")):
            from services import open_aspect as open_aspect_module

            open_aspect_module._nlp.cache_clear()
            with self.assertRaises(RuntimeError):
                extract_open_aspects(self.SAMPLE_TEXT, max_aspects=8)


if __name__ == "__main__":
    unittest.main()
