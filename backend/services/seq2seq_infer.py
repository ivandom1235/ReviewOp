# proto/backend/services/seq2seq_infer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from core.config import settings


VALID = {"positive", "neutral", "negative"}
_LABELS = ("positive", "neutral", "negative")


def _clean_label(s: str) -> str:
    s = (s or "").strip().lower()
    # strip common junk/punct the model may emit: "negative.", "Sentiment: negative", etc.
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    # take first valid token if present
    for tok in s.split():
        if tok in VALID:
            return tok
    return ""


@dataclass
class Seq2SeqEngine:
    tokenizer: Any
    model: Any
    device: torch.device

    @classmethod
    def load(cls) -> "Seq2SeqEngine":
        device = torch.device("cpu")
        tok = AutoTokenizer.from_pretrained(settings.seq2seq_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.seq2seq_model_name)
        model.to(device)
        model.eval()
        return cls(tokenizer=tok, model=model, device=device)

    def _prompt(self, evidence_text: str, aspect: str) -> str:
        evidence_text = (evidence_text or "").strip()
        aspect = (aspect or "").strip()

        # IMPORTANT: evidence_text should be the evidence sentence/snippet for the aspect,
        # not the entire review. Passing the whole review commonly flips everything to negative
        # when there's a "however/but" clause elsewhere.
        return (
            "Aspect-based sentiment classification.\n"
            "Return ONLY one word: positive, neutral, or negative.\n"
            "No punctuation. No explanation.\n\n"
            f"Aspect: {aspect}\n"
            f"Text: {evidence_text}\n"
            "Sentiment:"
        )

    def classify_sentiment(self, evidence_text: str, aspect: str) -> str:
        """
        Backward-compatible: returns only label.
        Pass evidence_text (sentence/snippet) NOT full review for higher accuracy.
        """
        label, _conf = self.classify_sentiment_with_confidence(evidence_text, aspect)
        return label

    def classify_sentiment_with_confidence(self, evidence_text: str, aspect: str) -> Tuple[str, float]:
        """
        Robust scoring via candidate likelihood (no reliance on generation parsing).
        Returns (label, confidence).
        """
        prompt = self._prompt(evidence_text, aspect)

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Score each candidate label by negative log-likelihood under the model.
        # Lower loss => better.
        losses: Dict[str, float] = {}
        with torch.no_grad():
            for lab in _LABELS:
                # target is the label token(s)
                y = self.tokenizer(
                    lab,
                    return_tensors="pt",
                    truncation=True,
                    max_length=8,
                ).input_ids.to(self.device)

                out = self.model(**enc, labels=y)
                loss = float(out.loss.item())
                losses[lab] = loss

        # Convert losses to probabilities (softmax over negative losses)
        neg = torch.tensor([-losses[l] for l in _LABELS], dtype=torch.float32)
        probs = torch.softmax(neg, dim=0).tolist()

        best_i = int(torch.tensor(probs).argmax().item())
        best_label = _LABELS[best_i]
        conf = float(probs[best_i])

        if best_label not in VALID:
            return "neutral", 0.0
        return best_label, conf

    def classify_sentiment_by_generate(self, evidence_text: str, aspect: str) -> str:
        """
        Optional: original generate-based approach, but with better output cleaning.
        """
        prompt = self._prompt(evidence_text, aspect)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(settings.seq2seq_max_new_tokens),
                do_sample=False,
                num_beams=4,
                early_stopping=True,
            )

        raw = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        lab = _clean_label(raw)
        return lab if lab in VALID else "neutral"