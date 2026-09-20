from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
REFERENCE_WEIGHTS = (0.6, 0.6, 0.4, 0.4)


def load_prediction_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"logits", "labels", "sample_ids"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Prediction archive {path} misses: {sorted(missing)}")
        return {key: archive[key].copy() for key in archive.files}


def fuse_prediction_archives(
    paths: Sequence[str | Path], weights: Sequence[float] = REFERENCE_WEIGHTS
) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one prediction archive is required")
    if len(paths) != len(weights):
        raise ValueError(f"Expected one weight per archive, got {len(weights)} for {len(paths)}")
    archives = [load_prediction_archive(path) for path in paths]
    reference = archives[0]
    labels = reference["labels"]
    sample_ids = reference["sample_ids"]
    logits_shape = reference["logits"].shape
    for path, archive in zip(paths[1:], archives[1:], strict=True):
        if archive["logits"].shape != logits_shape:
            raise ValueError(f"Logit shape mismatch in {path}: {archive['logits'].shape}")
        if not np.array_equal(archive["labels"], labels):
            raise ValueError(f"Label ordering mismatch in {path}")
        if not np.array_equal(archive["sample_ids"], sample_ids):
            raise ValueError(f"Sample ordering mismatch in {path}")
    fused = np.zeros(logits_shape, dtype=np.float64)
    component_logits: dict[str, np.ndarray] = {}
    for index, (archive, weight) in enumerate(zip(archives, weights, strict=True)):
        logits = archive["logits"].astype(np.float64, copy=False)
        fused += float(weight) * logits
        name = REFERENCE_STREAMS[index] if index < len(REFERENCE_STREAMS) else f"stream_{index}"
        component_logits[name] = logits
    return {
        "logits": fused.astype(np.float32),
        "labels": labels,
        "sample_ids": sample_ids,
        "weights": np.asarray(weights, dtype=np.float32),
        "component_logits": component_logits,
    }
