from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.optim import AdamW

try:
    from .config import ProtonetConfig
    from .evaluator import evaluate_episodes
    from .model import ProtoNetModel
    from .progress import announce, task_bar
    from .quality_signals import example_quality_weight
    from .prototype_bank import PrototypeBank, build_global_prototype_bank
    from .training_utils import (
        clear_embedding_cache_if_trainable as _clear_embedding_cache_if_trainable,
        collect_unique_items as _collect_unique_items,
        composite_selection_score as _composite_selection_score,
        joint_label_from_item as _joint_label_from_item,
        build_offline_embedding_cache as _build_offline_embedding_cache,
        warmup_batch_size as _warmup_batch_size,
        warmup_label_cap as _warmup_label_cap,
    )
except ImportError:
    from config import ProtonetConfig
    from evaluator import evaluate_episodes
    from model import ProtoNetModel
    from progress import announce, task_bar
    from quality_signals import example_quality_weight
    from prototype_bank import PrototypeBank, build_global_prototype_bank
    from training_utils import (
        clear_embedding_cache_if_trainable as _clear_embedding_cache_if_trainable,
        collect_unique_items as _collect_unique_items,
        composite_selection_score as _composite_selection_score,
        joint_label_from_item as _joint_label_from_item,
        build_offline_embedding_cache as _build_offline_embedding_cache,
        warmup_batch_size as _warmup_batch_size,
        warmup_label_cap as _warmup_label_cap,
    )


@dataclass
class TrainingResult:
    model: ProtoNetModel
    history: List[Dict[str, Any]]
    checkpoint_path: Path
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    val_predictions: List[Dict[str, Any]]
    test_predictions: List[Dict[str, Any]]
    prototype_bank: PrototypeBank


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    items: List[Dict[str, Any]],
    temperature: float,
) -> torch.Tensor:
    if len(items) < 2:
        return embeddings.new_tensor(0.0)

    # 1. Build soft positive mask
    # Similarity = 1.0 for exact label match
    # Similarity = 0.5 for same latent_family match
    # Similarity = 0.0 otherwise
    num_items = len(items)
    mask = torch.zeros((num_items, num_items), device=embeddings.device, dtype=embeddings.dtype)
    
    for i in range(num_items):
        item_i = items[i]
        label_i = str(item_i.get("joint_label") or item_i.get("aspect") or "")
        family_i = str(item_i.get("latent_family") or "unknown")
        
        for j in range(num_items):
            if i == j:
                continue
            item_j = items[j]
            label_j = str(item_j.get("joint_label") or item_j.get("aspect") or "")
            family_j = str(item_j.get("latent_family") or "unknown")
            
            if label_i == label_j and label_i != "":
                mask[i, j] = 1.0
            elif family_i == family_j and family_i != "unknown":
                mask[i, j] = 0.5

    if mask.sum() == 0:
        return embeddings.new_tensor(0.0)

    # 2. Compute similarity matrix
    # embeddings should be normalized
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    
    # 3. Compute weighted SupCon loss
    # exp(sim)
    exp_sim = torch.exp(sim - torch.max(sim, dim=1, keepdim=True)[0])
    
    # Denominator: sum of all exp(sim) except self
    # We can use mask=torch.eye to zero out self-similarity
    self_mask = torch.eye(num_items, device=embeddings.device)
    denom = (exp_sim * (1 - self_mask)).sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    # log_prob = sim - log(denom)
    log_prob = (sim - torch.max(sim, dim=1, keepdim=True)[0]) - torch.log(denom)
    
    # Weighted average of log_prob for positives
    weighted_log_prob = (mask * log_prob).sum(dim=1)
    weights_sum = mask.sum(dim=1).clamp(min=1e-8)
    
    # Final loss is negative mean of weighted log probs where we have positives
    has_positives = mask.sum(dim=1) > 0
    if not has_positives.any():
        return embeddings.new_tensor(0.0)
        
    loss = - (weighted_log_prob[has_positives] / weights_sum[has_positives]).mean()
    return loss


def _use_cuda_amp(cfg: ProtonetConfig) -> bool:
    return bool(cfg.use_amp and cfg.device.type == "cuda")


def _encode_items_in_batches(
    model: ProtoNetModel,
    items: List[Dict[str, Any]],
    *,
    batch_size: int,
    desc: str,
    enabled: bool,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    total_chunks = max(1, (len(items) + batch_size - 1) // batch_size)
    with task_bar(total=total_chunks, desc=desc, enabled=enabled) as bar:
        for chunk_index, index in enumerate(range(0, len(items), batch_size), start=1):
            chunk = items[index : index + batch_size]
            embeddings.append(model.encode_items(chunk))
            bar.update(1)
            bar.set_postfix(batch=f"{chunk_index}/{total_chunks}", items=len(chunk))
    return torch.cat(embeddings, dim=0)


def _warmup_representations(model: ProtoNetModel, cfg: ProtonetConfig, optimizer: torch.optim.Optimizer, episodes: List[Dict[str, Any]]) -> None:
    if cfg.warmup_epochs <= 0:
        return
    raw_items = _collect_unique_items(episodes)
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_items:
        by_label.setdefault(_joint_label_from_item(item, cfg.joint_label_separator), []).append(item)
    items: List[Dict[str, Any]] = []
    per_label_cap = _warmup_label_cap(model)
    for label in sorted(by_label):
        items.extend(by_label[label][: min(per_label_cap, len(by_label[label]))])
    if len(items) < 4:
        return
    labels = [_joint_label_from_item(item, cfg.joint_label_separator) for item in items]
    batch_size = _warmup_batch_size(model)
    for epoch in range(1, cfg.warmup_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = _encode_items_in_batches(
            model,
            items,
            batch_size=batch_size,
            desc=f"warmup:{epoch}/{cfg.warmup_epochs}",
            enabled=cfg.progress_enabled,
        )
        loss = _supervised_contrastive_loss(embeddings, items, temperature=cfg.contrastive_temperature)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _trainable_parameter_groups(model: ProtoNetModel, cfg: ProtonetConfig) -> List[Dict[str, Any]]:
    # Regularization: exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.weight"]

    def get_groups(params: Any, lr: float):
        decay_params = []
        no_decay_params = []
        for n, p in params:
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {"params": decay_params, "weight_decay": cfg.weight_decay, "lr": lr},
            {"params": no_decay_params, "weight_decay": 0.0, "lr": lr},
        ]

    groups = get_groups(model.projection.named_parameters(), cfg.learning_rate)
    # Add log_temperature to no_decay group of projection
    groups[1]["params"].append(model.log_temperature)

    if model.encoder.backend == "transformer" and model.encoder.model is not None:
        # Backbone parameters (optional)
        if model.encoder.trainable:
            groups.extend(get_groups(model.encoder.model.named_parameters(), cfg.encoder_learning_rate))
        
        # Head parameters (always trained if using transformer backbone)
        if model.encoder.pooling_head is not None:
            groups.extend(get_groups(model.encoder.pooling_head.named_parameters(), cfg.learning_rate))

    return [g for g in groups if g["params"]]


def _episode_loss(model: ProtoNetModel, episode: Dict[str, Any], cfg: ProtonetConfig) -> tuple[torch.Tensor, float]:
    return _episode_loss_with_weights(model, episode, cfg, {})


def _class_balanced_ce_weights(train_episodes: List[Dict[str, Any]], cfg: ProtonetConfig, beta: float = 0.9999) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for episode in train_episodes:
        for item in list(episode.get("query_set", [])) + list(episode.get("support_set", [])):
            label = _joint_label_from_item(item, cfg.joint_label_separator)
            counts[label] = counts.get(label, 0.0) + example_quality_weight(item)
    if not counts:
        return {}
    raw: Dict[str, float] = {}
    for label, n in counts.items():
        effective = (1.0 - (beta ** max(1.0, n))) / max(1e-8, 1.0 - beta)
        raw[label] = 1.0 / max(1e-8, effective)
    scale = sum(raw.values()) / max(1, len(raw))
    return {label: float(value / max(1e-8, scale)) for label, value in raw.items()}


def _focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=alpha)
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=focal_loss.device, dtype=focal_loss.dtype)
        focal_loss = focal_loss * sample_weights
        return focal_loss.sum() / sample_weights.sum().clamp(min=1e-6)
    return focal_loss.mean()


def _episode_loss_with_weights(
    model: ProtoNetModel,
    episode: Dict[str, Any],
    cfg: ProtonetConfig,
    class_weight_lookup: Dict[str, float],
) -> tuple[torch.Tensor, float]:
    use_amp = _use_cuda_amp(cfg)
    query_weights = torch.tensor(
        [example_quality_weight(item) for item in list(episode.get("query_set", []))],
        dtype=torch.float32,
        device=cfg.device,
    )
    with autocast(device_type="cuda", enabled=use_amp):
        out = model.episode_forward(episode)
        ce_weights = torch.ones(len(out.ordered_labels), device=out.logits.device, dtype=out.logits.dtype)
        for idx, label in enumerate(out.ordered_labels):
            ce_weights[idx] = float(class_weight_lookup.get(label, 1.0))
        
        loss = _focal_loss(
            out.logits,
            out.targets,
            alpha=ce_weights,
            gamma=float(cfg.focal_gamma),
            sample_weights=query_weights if len(query_weights) else None,
        )
        
        if cfg.contrastive_weight > 0:
            embeddings = torch.cat([out.support_embeddings, out.query_embeddings], dim=0)
            items = list(episode.get("support_set", [])) + list(episode.get("query_set", []))
            loss = loss + cfg.contrastive_weight * _supervised_contrastive_loss(embeddings, items, temperature=cfg.contrastive_temperature)
            
        if cfg.ortho_weight > 0:
            # Orthogonal penalty between prototypes to enhance feature discriminativity
            similarity = torch.matmul(out.prototypes, out.prototypes.T)
            eye = torch.eye(out.prototypes.size(0), device=out.prototypes.device)
            ortho_loss = torch.norm(similarity - eye, p='fro')
            loss = loss + cfg.ortho_weight * ortho_loss

    preds = out.probabilities.argmax(dim=-1).tolist()
    targets = out.targets.detach().cpu().tolist()
    correct = sum(int(p == t) for p, t in zip(preds, targets))
    accuracy = correct / max(1, len(targets))
    return loss, accuracy


def _save_checkpoint(model: ProtoNetModel, cfg: ProtonetConfig, history: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_info": model.encoder.export_info(),
            "history": history,
            "config": cfg.to_dict(),
        },
        path,
    )


def load_checkpoint(model: ProtoNetModel, checkpoint_path: Path) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=model.cfg.device)
    model.load_state_dict(state["model_state_dict"])
    return state


def train_model(cfg: ProtonetConfig, episodes_by_split: Dict[str, List[Dict[str, Any]]]) -> TrainingResult:
    model = ProtoNetModel(cfg)
    optimizer = AdamW(_trainable_parameter_groups(model, cfg))
    scaler = GradScaler("cuda", enabled=_use_cuda_amp(cfg))

    history: List[Dict[str, Any]] = []
    best_val = float("-inf")
    wait = 0
    checkpoint_path = cfg.checkpoint_dir / "best.pt"
    train_episodes = episodes_by_split["train"]
    embedding_cache_summary = _build_offline_embedding_cache(model, episodes_by_split, cfg)
    announce(f"[train] embedding_cache={embedding_cache_summary}")
    class_weight_lookup = _class_balanced_ce_weights(train_episodes, cfg)
    _warmup_representations(model, cfg, optimizer, train_episodes)

    for epoch in range(1, cfg.epochs + 1):
        announce(f"\n[train] Epoch {epoch}/{cfg.epochs} | Processing {len(train_episodes)} episodes...")
        _clear_embedding_cache_if_trainable(model)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        recent_loss: deque[float] = deque(maxlen=20)
        recent_acc: deque[float] = deque(maxlen=20)
        optimizer_updates = 0
        optimizer.zero_grad(set_to_none=True)
        
        desc = f"train:{epoch}/{cfg.epochs}"
        with task_bar(total=len(train_episodes), desc=desc, enabled=cfg.progress_enabled) as bar:
            bar.set_postfix(status="initializing...")
            for step_index, episode in enumerate(train_episodes, start=1):
                loss, accuracy = _episode_loss_with_weights(model, episode, cfg, class_weight_lookup)
                scaled_loss = loss / max(1, cfg.gradient_accumulation_steps)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step_index % cfg.gradient_accumulation_steps == 0 or step_index == len(train_episodes):
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer_updates += 1
                    optimizer.zero_grad(set_to_none=True)

                current_loss = float(loss.detach().cpu().item())
                running_loss += current_loss
                running_acc += accuracy
                recent_loss.append(current_loss)
                recent_acc.append(accuracy)
                bar.update(1)
                bar.set_postfix(
                    loss=f"{current_loss:.3f}",
                    avg_loss=f"{running_loss / step_index:.3f}",
                    avg_acc=f"{running_acc / step_index:.3f}",
                    recent_loss=f"{sum(recent_loss) / len(recent_loss):.3f}",
                    recent_acc=f"{sum(recent_acc) / len(recent_acc):.3f}",
                    updates=optimizer_updates,
                )

        train_metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_episodes)),
            "train_accuracy": running_acc / max(1, len(train_episodes)),
        }
        val_metrics, _ = evaluate_episodes(model, episodes_by_split["val"], cfg, "val", include_predictions=False)
        train_metrics.update(
            {
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        history.append(train_metrics)
        announce(f"[report] Epoch {epoch} summary: loss={train_metrics['train_loss']:.4f}, acc={train_metrics['train_accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        selection_score = _composite_selection_score(val_metrics)
        if selection_score > best_val:
            best_val = selection_score
            wait = 0
            _save_checkpoint(model, cfg, history, checkpoint_path)
            announce(f"[train] new best checkpoint composite_score={best_val:.3f}")
        else:
            wait += 1
            if wait >= cfg.patience:
                announce(f"[train] early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    load_checkpoint(model, checkpoint_path)
    _clear_embedding_cache_if_trainable(model)
    
    val_metrics, val_predictions = evaluate_episodes(model, episodes_by_split["val"], cfg, "val")
    test_metrics, test_predictions = evaluate_episodes(model, episodes_by_split["test"], cfg, "test")
    
    protocol_metrics: Dict[str, Any] = {}
    for key, episodes in episodes_by_split.items():
        if key in {"train", "val", "test"}:
            continue
        m, _ = evaluate_episodes(model, episodes, cfg, key, include_predictions=False)
        protocol_metrics[key] = m

    if protocol_metrics:
        val_metrics["protocol_full_eval"] = {k: v for k, v in protocol_metrics.items() if k.startswith("val")}
        test_metrics["protocol_full_eval"] = {k: v for k, v in protocol_metrics.items() if k.startswith("test")}
        if "domain_holdout" in protocol_metrics:
            test_metrics["domain_holdout_eval"] = protocol_metrics["domain_holdout"]

    prototype_bank = build_global_prototype_bank(model, episodes_by_split["train"], cfg)

    history_path = cfg.output_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return TrainingResult(
        model=model,
        history=history,
        checkpoint_path=checkpoint_path,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        prototype_bank=prototype_bank,
    )
