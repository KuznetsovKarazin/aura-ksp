from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def checkpoint_rank(metrics: dict[str, Any], epoch: int) -> tuple[float, float, float, int]:
    """Frozen checkpoint rule: Top-1, Macro-F1, lower NLL, then earlier epoch."""
    return (
        float(metrics["top1"]),
        float(metrics["macro_f1"]),
        -float(metrics["nll"]),
        -int(epoch),
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    name = str(train_cfg.get("optimizer", "sgd")).lower()
    common = {
        "lr": float(train_cfg["learning_rate"]),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            momentum=float(train_cfg.get("momentum", 0.9)),
            nesterov=bool(train_cfg.get("nesterov", True)),
            **common,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    name = str(config["training"].get("scheduler", "cosine")).lower()
    epochs = int(config["training"]["epochs"])
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    if name == "warmup_multistep":
        warmup_epochs = int(config["training"].get("warmup_epochs", 5))
        milestones = tuple(int(value) for value in config["training"].get("milestones", []))
        gamma = float(config["training"].get("lr_decay_rate", 0.1))

        def lr_factor(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            decay_count = sum(epoch >= milestone for milestone in milestones)
            return gamma**decay_count

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    accumulation: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    samples = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        inputs = batch["skeleton"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(inputs)
            loss = criterion(logits, labels) / accumulation
        scaler.scale(loss).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += float(loss.detach()) * accumulation * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        samples += labels.numel()
    return {
        "loss": loss_sum / max(1, samples),
        "top1": correct / max(1, samples),
        "seconds": time.perf_counter() - started,
    }
