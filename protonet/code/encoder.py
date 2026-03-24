from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except Exception:  # pragma: no cover
    HashingVectorizer = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

try:
    from .config import ProtonetConfig
except ImportError:
    from config import ProtonetConfig


SPECIAL_TOKENS = ["[E_START]", "[E_END]"]


def _format_transformer_load_error(model_name: str, exc: Exception, *, allow_model_download: bool) -> str:
    detail = str(exc).strip()
    base = f"Unable to load transformer encoder '{model_name}' from local cache."
    if allow_model_download:
        base = f"Unable to load transformer encoder '{model_name}'."
    lowered = detail.lower()
    if "sentencepiece" in lowered:
        return (
            f"{base} The tokenizer requires the `sentencepiece` package in this environment. "
            "Install it with `pip install sentencepiece` and rerun the command."
        )
    if "protobuf" in lowered:
        return (
            f"{base} The tokenizer requires the `protobuf` package in this environment. "
            "Install it with `pip install protobuf` and rerun the command."
        )
    if not allow_model_download:
        return (
            f"{base} If this model is not cached yet, rerun with `--allow-model-download` "
            "to let protonet fetch it from Hugging Face."
        )
    if detail:
        return f"{base} Original error: {detail}"
    return base


def format_input_text(review_text: str, evidence_text: str, domain: str) -> str:
    review = str(review_text or "").strip()
    evidence = str(evidence_text or "").strip()
    dom = str(domain or "unknown").strip().lower() or "unknown"
    if review and evidence and evidence in review:
        marked = review.replace(evidence, f"[E_START] {evidence} [E_END]", 1)
    elif review:
        marked = f"{review} Evidence: [E_START] {evidence or review} [E_END]"
    else:
        marked = f"[E_START] {evidence} [E_END]"
    return f"[DOMAIN={dom}] {marked}".strip()


class HybridTextEncoder(nn.Module):
    def __init__(self, cfg: ProtonetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = "bow"
        self.warning: str | None = None
        self.model_name = cfg.encoder_model_name
        self.trainable = False
        self.hidden_size = cfg.bow_dim
        self.tokenizer = None
        self.model = None
        self.vectorizer = None

        backend = cfg.encoder_backend.lower().strip()
        wants_transformer = backend in {"auto", "transformer"}
        if wants_transformer and AutoTokenizer is not None and AutoModel is not None:
            try:
                local_files_only = not cfg.allow_model_download
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model_name, local_files_only=local_files_only)
                self.model = AutoModel.from_pretrained(cfg.encoder_model_name, local_files_only=local_files_only)
                added = self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
                if added:
                    self.model.resize_token_embeddings(len(self.tokenizer))
                self.backend = "transformer"
                self.hidden_size = int(self.model.config.hidden_size)
                self.trainable = bool(cfg.train_encoder)
                if not self.trainable:
                    for param in self.model.parameters():
                        param.requires_grad = False
            except Exception as exc:
                if backend == "transformer" and cfg.strict_encoder:
                    raise RuntimeError(
                        _format_transformer_load_error(
                            cfg.encoder_model_name,
                            exc,
                            allow_model_download=cfg.allow_model_download,
                        )
                    ) from exc
                self.warning = (
                    f"{_format_transformer_load_error(cfg.encoder_model_name, exc, allow_model_download=cfg.allow_model_download)} "
                    "Falling back to hashed bag-of-words features."
                )

        if self.backend != "transformer":
            if HashingVectorizer is None:
                raise RuntimeError("scikit-learn is required for the bag-of-words fallback encoder")
            self.vectorizer = HashingVectorizer(
                n_features=cfg.bow_dim,
                norm="l2",
                alternate_sign=False,
                ngram_range=(1, 2),
            )
            self.backend = "bow"
            self.trainable = False
            self.hidden_size = cfg.bow_dim

    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.backend == "transformer":
            return self._encode_transformer(texts)
        return self._encode_bow(texts)

    def _encode_bow(self, texts: List[str]) -> torch.Tensor:
        assert self.vectorizer is not None
        matrix = self.vectorizer.transform(texts)
        dense = matrix.toarray()
        return torch.tensor(dense, dtype=torch.float32, device=self.cfg.device)

    def _encode_transformer(self, texts: List[str]) -> torch.Tensor:
        assert self.tokenizer is not None and self.model is not None
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.cfg.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        start_id = self.tokenizer.convert_tokens_to_ids("[E_START]")
        end_id = self.tokenizer.convert_tokens_to_ids("[E_END]")

        pooled: List[torch.Tensor] = []
        for row_index in range(hidden.size(0)):
            input_ids = encoded["input_ids"][row_index]
            start_pos = (input_ids == start_id).nonzero(as_tuple=False)
            end_pos = (input_ids == end_id).nonzero(as_tuple=False)
            if len(start_pos) and len(end_pos):
                start = int(start_pos[0].item()) + 1
                end = int(end_pos[0].item())
                if start < end:
                    pooled.append(hidden[row_index, start:end].mean(dim=0))
                    continue
            mask = attention_mask[row_index].unsqueeze(-1)
            pooled.append((hidden[row_index] * mask).sum(dim=0) / mask.sum().clamp(min=1))
        return torch.stack(pooled, dim=0)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)

    def export_info(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "trainable": self.trainable,
            "special_tokens": SPECIAL_TOKENS,
            "warning": self.warning,
        }

    def export_state(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"backend": self.backend}
        if self.backend == "transformer" and self.model is not None:
            payload["state_dict"] = self.model.state_dict()
        return payload
