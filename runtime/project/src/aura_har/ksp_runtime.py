from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aura_har.adaptive_fusion import fuse_logits, normalized_prior
from aura_har.data.sparse_reference import SparseReferenceDataset
from aura_har.ksp_execution import (
    ksp_torch_load,
)
from aura_har.ksp_protocol import FIXED_PRIOR, STREAMS, validate_run_manifest_provenance
from aura_har.models import build_model
from aura_har.utils.reproducibility import seed_worker


def build_ksp_loader(
    config: dict[str, Any], split: str, seed: int, shuffle: bool | None = None
) -> tuple[SparseReferenceDataset, DataLoader]:
    """Build the isolated train-only sparse loader without touching legacy runtime code."""
    if split not in {"train", "heldout", "val"}:
        raise ValueError("AURA-KSP exposes only train and setup-held-out loaders")
    validate_run_manifest_provenance(config, split=split)
    data = config["data"]
    dataset = SparseReferenceDataset(
        manifest=data["manifest"],
        store_metadata=data["sparse_store_root"],
        num_frames=int(data["num_frames"]),
        train=split == "train",
        seed=int(data.get("sampling_seed", seed)),
        modality=str(data["modality"]),
        temporal_mode=str(data["temporal_mode"]),
        p_interval_train=data.get("p_interval_train", [0.5, 1.0]),
        p_interval_eval=data.get("p_interval_eval", [0.95]),
        random_rotation_enabled=bool(data.get("random_rotation", True)),
        rotation_theta=float(data.get("rotation_theta", 0.3)),
        train_random_offset=bool(data.get("train_random_offset", True)),
        return_streams=False,
    )
    section = config["training"] if split == "train" else config.get("evaluation", {})
    batch_size = int(section.get("batch_size", config["training"]["batch_size"]))
    num_workers = int(section.get("num_workers", 0))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else bool(shuffle),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    return dataset, loader


@torch.inference_mode()
def collect_ksp_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[str] = []
    indices: list[np.ndarray] = []
    setups: list[np.ndarray] = []
    acquisition: dict[str, list[np.ndarray]] = {
        "source_indices": [],
        "source_index_count": [],
        "left_gather": [],
        "right_gather": [],
        "right_weight": [],
        "source_bytes_logical": [],
    }
    for batch in loader:
        output = model(batch["skeleton"].to(device, non_blocking=True))
        logits.append(output.cpu().numpy())
        labels.append(batch["label"].numpy())
        sample_ids.extend(batch["sample_id"])
        indices.append(batch["indices"].numpy())
        setups.append(batch["setup"].numpy())
        for name, values in acquisition.items():
            value = batch[name]
            values.append(
                value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            )
    return (
        np.concatenate(logits),
        np.concatenate(labels),
        sample_ids,
        np.concatenate(indices),
        np.concatenate(setups),
        {name: np.concatenate(values) for name, values in acquisition.items()},
    )


class FourStreamModelBundle(nn.Module):
    """Execute the four separately trained stream models after one shared source read."""

    def __init__(self, models: Mapping[str, nn.Module]) -> None:
        super().__init__()
        if tuple(models) != STREAMS:
            raise ValueError(f"Expected ordered stream bundle {STREAMS}, got {tuple(models)}")
        self.models = nn.ModuleDict(dict(models))

    def forward(self, streams: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(streams) != STREAMS:
            raise ValueError(f"Expected ordered stream tensors {STREAMS}, got {tuple(streams)}")
        component = torch.stack(
            [self.models[name](streams[name]) for name in STREAMS], dim=1
        )
        weights = torch.as_tensor(
            FIXED_PRIOR, dtype=component.dtype, device=component.device
        ).view(1, len(STREAMS), 1)
        fused = (component * weights).sum(dim=1)
        return fused, component


def load_four_stream_bundle(
    configs: Mapping[str, dict[str, Any]],
    checkpoints: Mapping[str, str | Path],
    device: torch.device,
) -> FourStreamModelBundle:
    if tuple(configs) != STREAMS or tuple(checkpoints) != STREAMS:
        raise ValueError("Four ordered configs and checkpoints are required")
    models: dict[str, nn.Module] = {}
    for stream in STREAMS:
        config = configs[stream]
        if str(config["data"]["modality"]) != stream:
            raise ValueError(f"Config modality mismatch for {stream}")
        model = build_model(config).to(device)
        payload = ksp_torch_load(checkpoints[stream], map_location=device, weights_only=False)
        model.load_state_dict(payload.get("model", payload))
        model.eval()
        models[stream] = model
    return FourStreamModelBundle(models).to(device)


def fuse_fixed_prior_numpy(stream_logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(stream_logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[1] != len(STREAMS):
        raise ValueError(f"Expected N,{len(STREAMS)},C logits, got {logits.shape}")
    return fuse_logits(logits, normalized_prior(FIXED_PRIOR))
