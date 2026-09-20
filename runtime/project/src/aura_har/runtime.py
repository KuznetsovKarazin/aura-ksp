from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from aura_har.data.dataset import build_dataset
from aura_har.models import build_model
from aura_har.utils.reproducibility import seed_worker


def select_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for name in ["torch", "numpy", "scikit-learn", "PyYAML", "pandas"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        metadata["gpu_model"] = torch.cuda.get_device_name(0)
        metadata["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return metadata


def build_loader(
    config: dict[str, Any], split: str, seed: int, shuffle: bool | None = None
) -> tuple[Any, DataLoader]:
    dataset = build_dataset(config, split, seed)
    section = config["training"] if split == "train" else config.get("evaluation", {})
    batch_size = int(section.get("batch_size", config["training"]["batch_size"]))
    num_workers = int(section.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        # Dataset epoch is updated in the parent process before each epoch. Recreating
        # workers propagates that state reliably on both spawn (Windows) and fork (Linux).
        persistent_workers=False,
    )
    return dataset, loader


@torch.inference_mode()
def collect_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ids: list[str] = []
    all_indices: list[np.ndarray] = []
    for batch in loader:
        inputs = batch["skeleton"].to(device, non_blocking=True)
        logits = model(inputs)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_ids.extend(batch["sample_id"])
        all_indices.append(batch["indices"].numpy())
    return (
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        all_ids,
        np.concatenate(all_indices),
    )


def load_model_checkpoint(
    config: dict[str, Any], checkpoint: str | Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_model(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    return model, payload
