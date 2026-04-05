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
    from .prototype_bank import PrototypeBank, build_global_prototype_bank
except ImportError:
    from config import ProtonetConfig
    from evaluator import evaluate_episodes
    from model import ProtoNetModel
    from progress import announce, task_bar
    from prototype_bank import PrototypeBank, build_global_prototype_bank


@dataclass
class TrainingResult:
    model: ProtoNetModel
    history: List[Dict[str, Any]]
    checkpoint_path: Path
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    prototype_bank: PrototypeBank


def _joint_label_from_item(item: Dict[str, Any], separator: str) -> str:
    label = item.get("joint_label")
    if label:
        return str(label)
    aspect = str(item.get("aspect") or item.get("implicit_aspect") or "unknown").strip()
    sentiment = str(item.get("sentiment") or "neutral").strip().lower() or "neutral"
    return f"{aspect}{separator}{sentiment}"


def _collect_unique_items(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    items: List[Dict[str, Any]] = []
    for episode in episodes:
        for item in list(episode.get("support_set", [])) + list(episode.get("query_set", [])):
            key = str(item.get("example_id") or item.get("parent_review_id"))
            if key in seen:
                continue
            seen.add(key)
            items.append(dict(item))
    return items


def _supervised_contrastive_loss(embeddings: torch.Tensor, labels: List[str], temperature: float = 0.2) -> torch.Tensor:
    if len(labels) < 2:
        return embeddings.new_tensor(0.0)
    label_tensor = torch.tensor([hash(label) % 10_000_019 for label in labels], device=embeddings.device)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    mask = torch.eye(similarity.size(0), device=embeddings.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, -1e9)
    positive_mask = (label_tensor.unsqueeze(0) == label_tensor.unsqueeze(1)) & ~mask
    if positive_mask.sum() == 0:
        return embeddings.new_tensor(0.0)
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.float().sum(dim=1).clamp(min=1.0)
    valid = positive_mask.any(dim=1)
    return -mean_log_prob_pos[valid].mean()


def _warmup_label_cap(model: ProtoNetModel) -> int:
    if model.encoder.backend == "transformer":
        if model.cfg.device == "cpu":
            return 4
        return 8
    return 12


def _warmup_batch_size(model: ProtoNetModel) -> int:
    if model.encoder.backend == "transformer":
        if model.cfg.device == "cpu":
            return 16
        return 48
    return 128


def _encode_items_in_batches(
    model: ProtoNetModel,
    items: List[Dict[str, Any]],
    *,
    batch_size: int,
    desc: str,
    enabled: bool,
) -> torch.Tensor:
    chunks = [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
    embeddings: List[torch.Tensor] = []
    with task_bar(total=len(chunks), desc=desc, enabled=enabled) as bar:
        for chunk_index, chunk in enumerate(chunks, start=1):
            embeddings.append(model.encode_items(chunk))
            bar.update(1)
            bar.set_postfix(batch=f"{chunk_index}/{len(chunks)}", items=len(chunk))
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
        loss = _supervised_contrastive_loss(embeddings, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _trainable_parameter_groups(model: ProtoNetModel, cfg: ProtonetConfig) -> List[Dict[str, Any]]:
    # Regularization: exclude bias and LayerNorm from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    
    def get_groups(module_name: str, params: Any, lr: float):
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

    groups = get_groups("projection", model.projection.named_parameters(), cfg.learning_rate)
    # Add log_temperature to no_decay group of projection
    groups[1]["params"].append(model.log_temperature)
    
    if model.encoder.backend == "transformer" and model.encoder.trainable and model.encoder.model is not None:
        groups.extend(get_groups("encoder", model.encoder.model.named_parameters(), cfg.encoder_learning_rate))
    
    return [g for g in groups if g["params"]]


def _episode_loss(model: ProtoNetModel, episode: Dict[str, Any], cfg: ProtonetConfig) -> tuple[torch.Tensor, float]:
    use_amp = cfg.use_amp and cfg.device == "cuda"
    with autocast(device_type="cuda", enabled=use_amp):
        out = model.episode_forward(episode)
        loss = F.cross_entropy(out.logits, out.targets)
        if cfg.contrastive_weight > 0:
            combined = list(episode.get("support_set", [])) + list(episode.get("query_set", []))
            embeddings = model.encode_items(combined)
            labels = [_joint_label_from_item(item, cfg.joint_label_separator) for item in combined]
            loss = loss + cfg.contrastive_weight * _supervised_contrastive_loss(embeddings, labels)
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
    scaler = GradScaler("cuda", enabled=cfg.use_amp and cfg.device == "cuda")

    history: List[Dict[str, Any]] = []
    best_val = float("-inf")
    wait = 0
    checkpoint_path = cfg.checkpoint_dir / "best.pt"
    train_episodes = episodes_by_split["train"]
    _warmup_representations(model, cfg, optimizer, train_episodes)

    for epoch in range(1, cfg.epochs + 1):
        announce(f"\n[train] Epoch {epoch}/{cfg.epochs} | Processing {len(train_episodes)} episodes...")
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        recent_loss: deque[float] = deque(maxlen=20)
        recent_acc: deque[float] = deque(maxlen=20)
        optimizer_updates = 0
        optimizer.zero_grad(set_to_none=True)
        
        desc = f"train:{epoch}/{cfg.epochs}"
        with task_bar(total=len(train_episodes), desc=desc, enabled=cfg.progress_enabled) as bar:
            # Refresh bar immediately to show it's active even before the first episode finishes
            bar.set_postfix(status="initializing...")
            for step_index, episode in enumerate(train_episodes, start=1):
                if step_index == 1:
                    bar.set_postfix(status="running forward pass...")
                loss, accuracy = _episode_loss(model, episode, cfg)
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
        val_metrics, _ = evaluate_episodes(model, episodes_by_split["val"], cfg, "val")
        train_metrics.update(
            {
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        history.append(train_metrics)
        # Summary for researchers to track progress in terminal
        announce(f"[report] Epoch {epoch} summary: loss={train_metrics['train_loss']:.4f}, acc={train_metrics['train_accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f}")
        announce(
            f"[train] epoch {epoch} complete "
            f"loss={train_metrics['train_loss']:.3f} "
            f"acc={train_metrics['train_accuracy']:.3f} "
            f"val_acc={train_metrics['val_accuracy']:.3f} "
            f"val_f1={train_metrics['val_macro_f1']:.3f}"
        )

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            wait = 0
            _save_checkpoint(model, cfg, history, checkpoint_path)
            announce(f"[train] new best checkpoint val_acc={best_val:.3f}")
        else:
            wait += 1
            if wait >= cfg.patience:
                announce(f"[train] early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    load_checkpoint(model, checkpoint_path)
    val_metrics, _ = evaluate_episodes(model, episodes_by_split["val"], cfg, "val")
    test_metrics, _ = evaluate_episodes(model, episodes_by_split["test"], cfg, "test")
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
        prototype_bank=prototype_bank,
    )
