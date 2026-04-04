from __future__ import annotations

import argparse
import statistics
import time

import torch

from config import ProtonetConfig, seed_everything
from encoder import HybridTextEncoder


def _legacy_pool(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, start_id: int, end_id: int) -> torch.Tensor:
    pooled = []
    for row_index in range(hidden.size(0)):
        token_row = input_ids[row_index]
        start_pos = (token_row == start_id).nonzero(as_tuple=False)
        end_pos = (token_row == end_id).nonzero(as_tuple=False)
        if len(start_pos) and len(end_pos):
            start = int(start_pos[0].item()) + 1
            end = int(end_pos[0].item())
            if start < end:
                pooled.append(hidden[row_index, start:end].mean(dim=0))
                continue
        mask = attention_mask[row_index].unsqueeze(-1)
        pooled.append((hidden[row_index] * mask).sum(dim=0) / mask.sum().clamp(min=1))
    return torch.stack(pooled, dim=0)


def _encode_legacy(encoder: HybridTextEncoder, texts: list[str]) -> torch.Tensor:
    assert encoder.tokenizer is not None and encoder.model is not None
    encoded = encoder.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=encoder.cfg.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(encoder.cfg.device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = encoder.model(**encoded)
    hidden = outputs.last_hidden_state
    start_id = encoder.tokenizer.convert_tokens_to_ids("[E_START]")
    end_id = encoder.tokenizer.convert_tokens_to_ids("[E_END]")
    return _legacy_pool(
        hidden,
        encoded["input_ids"],
        encoded["attention_mask"],
        start_id,
        end_id,
    )


def _time_run(fn, repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Parity and speed benchmark for vectorized evidence pooling")
    parser.add_argument("--model-name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = ProtonetConfig(
        encoder_backend="transformer",
        encoder_model_name=args.model_name,
        strict_encoder=True,
        allow_model_download=bool(args.allow_model_download),
        train_encoder=False,
    )
    encoder = HybridTextEncoder(cfg).to(cfg.device)
    if encoder.backend != "transformer":
        print("FAIL: transformer backend is required for this benchmark")
        return 1

    texts = []
    for idx in range(args.batch_size):
        evidence = f"battery life dropped after update {idx}"
        if idx % 3 == 0:
            text = f"[DOMAIN=electronics] The phone worked fine but [E_START] {evidence} [E_END] and I am annoyed."
        elif idx % 3 == 1:
            text = f"[DOMAIN=electronics] I tested it for days and {evidence}."
        else:
            text = f"[DOMAIN=electronics] [E_START] display brightness is solid {idx} [E_END] and performance is stable."
        texts.append(text)

    with torch.no_grad():
        legacy = _encode_legacy(encoder, texts)
        vectorized = encoder._encode_transformer(texts)

    same_shape = tuple(legacy.shape) == tuple(vectorized.shape)
    same_dtype = legacy.dtype == vectorized.dtype
    max_abs = float((legacy - vectorized).abs().max().item())
    all_close = torch.allclose(legacy, vectorized, rtol=args.rtol, atol=args.atol)

    legacy_times = _time_run(lambda: _encode_legacy(encoder, texts), args.repeats)
    vectorized_times = _time_run(lambda: encoder._encode_transformer(texts), args.repeats)
    legacy_med = statistics.median(legacy_times)
    vector_med = statistics.median(vectorized_times)
    speedup = (legacy_med / vector_med) if vector_med > 0 else 0.0

    print(f"shape_match={same_shape} dtype_match={same_dtype} allclose={all_close} max_abs_diff={max_abs:.8f}")
    print(f"legacy_median_ms={legacy_med:.2f} vectorized_median_ms={vector_med:.2f} speedup={speedup:.2f}x")

    if not (same_shape and same_dtype and all_close):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
