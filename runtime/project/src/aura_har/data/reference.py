from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional

from aura_har.sampling import build_sampler

REFERENCE_MODALITIES = frozenset({"joint", "bone", "joint_motion", "bone_motion"})
NTU_BONE_PAIRS_1BASED = (
    (1, 2),
    (2, 21),
    (3, 21),
    (4, 3),
    (5, 21),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 21),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
    (22, 23),
    (21, 21),
    (23, 8),
    (24, 25),
    (25, 12),
)


def valid_crop_bounds(
    valid_length: int,
    p_interval: Sequence[float],
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Return the reference valid-frame crop without changing temporal resolution."""
    if valid_length <= 0:
        raise ValueError(f"valid_length must be positive, got {valid_length}")
    interval = tuple(float(value) for value in p_interval)
    if len(interval) not in {1, 2}:
        raise ValueError("p_interval must contain one evaluation value or two train bounds")
    if any(value <= 0 or value > 1 for value in interval):
        raise ValueError(f"p_interval values must be in (0, 1], got {interval}")

    if len(interval) == 1:
        proportion = interval[0]
        bias = int((1.0 - proportion) * valid_length / 2)
        start = bias
        stop = valid_length - bias
    else:
        low, high = interval
        if low > high:
            raise ValueError(f"p_interval lower bound exceeds upper bound: {interval}")
        proportion = float(rng.uniform(low, high))
        cropped_length = min(max(int(np.floor(valid_length * proportion)), 64), valid_length)
        start = int(rng.integers(0, valid_length - cropped_length + 1))
        stop = start + cropped_length
    if stop <= start:
        raise ValueError(f"Reference crop is empty: length={valid_length}, interval={interval}")
    return start, stop


def valid_crop_resize(
    sequence: np.ndarray,
    valid_length: int,
    p_interval: Sequence[float],
    window: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the official CTR-GCN valid-frame crop and bilinear resize."""
    if sequence.ndim != 4:
        raise ValueError(f"Expected C,T,V,M sequence, got {sequence.shape}")
    if valid_length <= 0 or valid_length > sequence.shape[1]:
        raise ValueError(f"valid_length must be in [1, {sequence.shape[1]}], got {valid_length}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    interval = tuple(float(value) for value in p_interval)
    start, stop = valid_crop_bounds(valid_length, interval, rng)
    cropped = np.ascontiguousarray(sequence[:, start:stop], dtype=np.float32)
    cropped_length = int(cropped.shape[1])
    if cropped_length <= 0:
        raise ValueError(f"Reference crop is empty: length={valid_length}, interval={interval}")

    channels, _, joints, persons = cropped.shape
    tensor = torch.from_numpy(cropped)
    tensor = tensor.permute(0, 2, 3, 1).contiguous().view(channels * joints * persons, -1)
    tensor = functional.interpolate(
        tensor[None, None],
        size=(channels * joints * persons, window),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    resized = (
        tensor.view(channels, joints, persons, window).permute(0, 3, 1, 2).contiguous().numpy()
    )
    indices = np.rint(np.linspace(start, stop - 1, window)).astype(np.int64)
    return resized, indices


def exact_temporal_sample(
    sequence: np.ndarray,
    valid_length: int,
    p_interval: Sequence[float],
    budget: int,
    sampler_name: str,
    rng: np.random.Generator,
    *,
    train: bool,
    train_random_offset: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool | str]]:
    """Crop and select exactly ``budget`` source observations without interpolation.

    Motion-guided selection inspects the complete valid crop. Uniform and random policies
    inspect only the returned source observations; compressed NPZ decoding remains outside
    this information-budget accounting and must be included in end-to-end benchmarks.
    """
    if sequence.ndim != 4:
        raise ValueError(f"Expected C,T,V,M sequence, got {sequence.shape}")
    if valid_length <= 0 or valid_length > sequence.shape[1]:
        raise ValueError(f"valid_length must be in [1, {sequence.shape[1]}], got {valid_length}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    start, stop = valid_crop_bounds(valid_length, p_interval, rng)
    cropped = np.ascontiguousarray(sequence[:, start:stop], dtype=np.float32)
    sampler = build_sampler(
        sampler_name,
        train=train,
        random_offset=train and train_random_offset,
    )
    local_indices = np.asarray(
        sampler(cropped, cropped.shape[1], int(budget), rng), dtype=np.int64
    )
    if local_indices.shape != (int(budget),):
        raise ValueError(
            f"Temporal sampler must return exactly {budget} indices, got {local_indices.shape}"
        )
    if np.any(local_indices < 0) or np.any(local_indices >= cropped.shape[1]):
        raise ValueError("Temporal sampler returned an out-of-crop index")
    name = str(sampler_name).lower()
    full_scan = name.startswith("motion")
    global_indices = local_indices + start
    metadata: dict[str, int | bool | str] = {
        "temporal_mode": "exact_k",
        "sampler": name,
        "temporal_budget": int(budget),
        "crop_start": int(start),
        "cropped_sequence_length": int(cropped.shape[1]),
        "source_frames_observed": (
            int(cropped.shape[1]) if full_scan else int(np.unique(local_indices).size)
        ),
        "selector_requires_full_sequence": full_scan,
    }
    return cropped, global_indices, metadata


def random_rotation(
    sequence: np.ndarray, rng: np.random.Generator, theta: float = 0.3
) -> np.ndarray:
    """Rotate all frames by one random XYZ rotation, matching the reference feeder."""
    if sequence.shape[0] != 3:
        raise ValueError(f"3D rotation requires three coordinate channels, got {sequence.shape}")
    angles = rng.uniform(-theta, theta, size=3).astype(np.float32)
    cos_r = np.cos(angles)
    sin_r = np.sin(angles)
    rx = np.asarray(
        [[1, 0, 0], [0, cos_r[0], sin_r[0]], [0, -sin_r[0], cos_r[0]]],
        dtype=np.float32,
    )
    ry = np.asarray(
        [[cos_r[1], 0, -sin_r[1]], [0, 1, 0], [sin_r[1], 0, cos_r[1]]],
        dtype=np.float32,
    )
    rz = np.asarray(
        [[cos_r[2], sin_r[2], 0], [-sin_r[2], cos_r[2], 0], [0, 0, 1]],
        dtype=np.float32,
    )
    rotation = rz @ ry @ rx
    return np.ascontiguousarray(np.einsum("ij,jtvm->itvm", rotation, sequence))


def bone_modality(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape[2] != 25:
        raise ValueError(f"NTU bone modality requires 25 joints, got {sequence.shape[2]}")
    output = np.zeros_like(sequence)
    for joint, parent in NTU_BONE_PAIRS_1BASED:
        output[:, :, joint - 1] = sequence[:, :, joint - 1] - sequence[:, :, parent - 1]
    return output


def motion_modality(sequence: np.ndarray) -> np.ndarray:
    output = np.zeros_like(sequence)
    output[:, :-1] = sequence[:, 1:] - sequence[:, :-1]
    return output


def apply_modality(sequence: np.ndarray, modality: str) -> np.ndarray:
    name = str(modality).lower()
    if name not in REFERENCE_MODALITIES:
        raise ValueError(f"Unknown reference modality: {modality}")
    output = sequence
    if name in {"bone", "bone_motion"}:
        output = bone_modality(output)
    if name in {"joint_motion", "bone_motion"}:
        output = motion_modality(output)
    return np.ascontiguousarray(output, dtype=np.float32)
