from __future__ import annotations

import os

from dataset_builder.explicit.spacy_pipeline import load_spacy


def test_spacy_loader_does_not_download(monkeypatch) -> None:
    def _fail_system(_cmd: str) -> int:
        raise AssertionError("os.system should not be called by load_spacy")

    monkeypatch.setattr(os, "system", _fail_system)
    nlp = load_spacy("missing_model_for_test_report4")
    assert "sentencizer" in nlp.pipe_names
