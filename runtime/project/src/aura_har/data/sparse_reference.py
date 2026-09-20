from __future__ import annotations

import hashlib
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aura_har.data.reference import (
    REFERENCE_MODALITIES,
    apply_modality,
    random_rotation,
    valid_crop_bounds,
)
from aura_har.utils.io import read_jsonl

from .sparse_store import FRAME_BYTES, NTU120FrameStore

FOUR_STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


@dataclass(frozen=True)
class SparseTemporalPlan:
    mode: str
    crop_start: int
    crop_stop: int
    output_source_indices: np.ndarray
    read_indices: np.ndarray
    left_gather: np.ndarray
    right_gather: np.ndarray
    right_weight: np.ndarray

    @property
    def output_frames(self) -> int:
        return int(self.output_source_indices.size)

    @property
    def source_frames_read(self) -> int:
        return int(self.read_indices.size)

    @property
    def cropped_sequence_length(self) -> int:
        return int(self.crop_stop - self.crop_start)


def _freeze_int_array(value: np.ndarray | Sequence[int]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.int64)
    result.setflags(write=False)
    return result


def _freeze_float_array(value: np.ndarray | Sequence[float]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float32)
    result.setflags(write=False)
    return result


def _unique_gather(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(np.asarray(indices, dtype=np.int64), return_inverse=True)
    return _freeze_int_array(unique), _freeze_int_array(inverse)


def plan_exact_k_uniform(
    valid_length: int,
    p_interval: Sequence[float],
    budget: int,
    rng: np.random.Generator,
    *,
    train: bool,
    train_random_offset: bool = True,
) -> SparseTemporalPlan:
    """Plan exact-K uniform retrieval without inspecting any skeleton coordinates."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    start, stop = valid_crop_bounds(valid_length, p_interval, rng)
    crop_length = stop - start
    random_offset = bool(train and train_random_offset)
    if crop_length < budget or not random_offset:
        local = np.rint(np.linspace(0, crop_length - 1, budget)).astype(np.int64)
    else:
        boundaries = np.linspace(0, crop_length, budget + 1, dtype=np.int64)
        local = np.asarray(
            [
                int(rng.integers(left, max(right, left + 1)))
                for left, right in pairwise(boundaries)
            ],
            dtype=np.int64,
        )
    output_indices = local + start
    read_indices, gather = _unique_gather(output_indices)
    zeros = np.zeros(budget, dtype=np.float32)
    return SparseTemporalPlan(
        mode="exact_k_uniform",
        crop_start=int(start),
        crop_stop=int(stop),
        output_source_indices=_freeze_int_array(output_indices),
        read_indices=read_indices,
        left_gather=gather,
        right_gather=gather,
        right_weight=_freeze_float_array(zeros),
    )


def plan_sparse_resize(
    valid_length: int,
    p_interval: Sequence[float],
    output_frames: int,
    rng: np.random.Generator,
) -> SparseTemporalPlan:
    """Plan align_corners=False temporal interpolation using only its source supports."""
    if output_frames <= 0:
        raise ValueError(f"output_frames must be positive, got {output_frames}")
    start, stop = valid_crop_bounds(valid_length, p_interval, rng)
    crop_length = stop - start
    positions = (np.arange(output_frames, dtype=np.float64) + 0.5) * (
        float(crop_length) / float(output_frames)
    ) - 0.5
    positions = np.clip(positions, 0.0, float(crop_length - 1))
    left_local = np.floor(positions).astype(np.int64)
    right_local = np.minimum(left_local + 1, crop_length - 1)
    weights = (positions - left_local).astype(np.float32)
    # Do not fetch a second frame when interpolation gives it exactly zero weight.
    right_local = np.where(weights == 0.0, left_local, right_local)
    left_global = left_local + start
    right_global = right_local + start
    read_indices = np.unique(np.concatenate([left_global, right_global]))
    left_gather = np.searchsorted(read_indices, left_global)
    right_gather = np.searchsorted(read_indices, right_global)
    nominal = np.rint(np.linspace(start, stop - 1, output_frames)).astype(np.int64)
    return SparseTemporalPlan(
        mode="sparse_resize",
        crop_start=int(start),
        crop_stop=int(stop),
        output_source_indices=_freeze_int_array(nominal),
        read_indices=_freeze_int_array(read_indices),
        left_gather=_freeze_int_array(left_gather),
        right_gather=_freeze_int_array(right_gather),
        right_weight=_freeze_float_array(weights),
    )


def plan_sparse_resize64(
    valid_length: int,
    p_interval: Sequence[float],
    rng: np.random.Generator,
) -> SparseTemporalPlan:
    return plan_sparse_resize(valid_length, p_interval, 64, rng)


def execute_sparse_plan(
    store: NTU120FrameStore,
    sample_id: str,
    plan: SparseTemporalPlan,
) -> np.ndarray:
    """Execute one temporal plan with exactly one source-store read."""
    supports = store.read_frames(sample_id, plan.read_indices)
    left = supports[:, plan.left_gather]
    if plan.mode == "exact_k_uniform":
        return np.ascontiguousarray(left, dtype=np.float32)
    if plan.mode != "sparse_resize":
        raise ValueError(f"Unsupported sparse temporal plan: {plan.mode}")
    right = supports[:, plan.right_gather]
    weight = plan.right_weight[None, :, None, None]
    resized = left * (np.float32(1.0) - weight) + right * weight
    return np.ascontiguousarray(resized, dtype=np.float32)


def construct_shared_streams(joint_sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Derive all four reference streams from one sampled and augmented joint tensor."""
    if joint_sequence.ndim != 4:
        raise ValueError(f"Expected C,T,V,M joint sequence, got {joint_sequence.shape}")
    return {stream: apply_modality(joint_sequence, stream) for stream in FOUR_STREAMS}


def _read_manifests(manifest: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    sources = [manifest] if isinstance(manifest, (str, Path)) else list(manifest)
    if not sources:
        raise ValueError("At least one sparse dataset manifest is required")
    rows = [row for source in sources for row in read_jsonl(source)]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Combined sparse manifests contain duplicate sample IDs")
    return rows


class SparseReferenceDataset(Dataset[dict[str, Any]]):
    """Train-only CTR-GCN dataset with source-index reads and shared stream construction."""

    def __init__(
        self,
        manifest: str | Path | Sequence[str | Path],
        store_metadata: str | Path,
        num_frames: int,
        train: bool,
        seed: int,
        modality: str,
        temporal_mode: str = "exact_k",
        p_interval_train: Sequence[float] = (0.5, 1.0),
        p_interval_eval: Sequence[float] = (0.95,),
        random_rotation_enabled: bool = True,
        rotation_theta: float = 0.3,
        sampler_name: str = "uniform",
        train_random_offset: bool = True,
        return_streams: bool = False,
    ) -> None:
        self.rows = _read_manifests(manifest)
        self.store = NTU120FrameStore(store_metadata)
        self.num_frames = int(num_frames)
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.modality = str(modality).lower()
        if self.modality not in REFERENCE_MODALITIES:
            raise ValueError(f"Unknown reference modality: {modality}")
        self.temporal_mode = str(temporal_mode).lower()
        if self.temporal_mode not in {"exact_k", "resize"}:
            raise ValueError(f"Unknown sparse temporal mode: {temporal_mode}")
        if self.temporal_mode == "resize" and self.num_frames != 64:
            raise ValueError("Sparse reference resize is frozen to the K64 baseline")
        self.p_interval_train = tuple(float(value) for value in p_interval_train)
        self.p_interval_eval = tuple(float(value) for value in p_interval_eval)
        self.random_rotation_enabled = bool(random_rotation_enabled)
        self.rotation_theta = float(rotation_theta)
        self.sampler_name = str(sampler_name).lower()
        if self.sampler_name != "uniform":
            raise ValueError("Sparse train-only protocol supports uniform sampling only")
        self.train_random_offset = bool(train_random_offset)
        self.return_streams = bool(return_streams)
        self._validate_rows_against_store()

    def _validate_rows_against_store(self) -> None:
        for row in self.rows:
            sample_id = str(row["sample_id"])
            record = self.store.record(sample_id)
            if int(row["label"]) != record.label:
                raise ValueError(f"Manifest/store label mismatch: {sample_id}")
            if "setup" in row and int(row["setup"]) != record.setup:
                raise ValueError(f"Manifest/store setup mismatch: {sample_id}")
            if (
                "sequence_length" in row
                and int(row["sequence_length"]) != record.sequence_length
            ):
                raise ValueError(f"Manifest/store sequence length mismatch: {sample_id}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _plan(self, length: int, rng: np.random.Generator) -> SparseTemporalPlan:
        interval = self.p_interval_train if self.train else self.p_interval_eval
        if self.temporal_mode == "resize":
            return plan_sparse_resize64(length, interval, rng)
        return plan_exact_k_uniform(
            length,
            interval,
            self.num_frames,
            rng,
            train=self.train,
            train_random_offset=self.train_random_offset,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = str(row["sample_id"])
        record = self.store.record(sample_id)
        stable = zlib.crc32(sample_id.encode("utf-8"))
        rng_seed = self.seed + stable + (self.epoch * 1_000_003 if self.train else 0)
        plan_rng = np.random.default_rng(rng_seed)
        rotation_rng = np.random.default_rng(rng_seed ^ 0x5DEECE66D)
        plan = self._plan(record.sequence_length, plan_rng)
        joint = execute_sparse_plan(self.store, sample_id, plan)
        if self.train and self.random_rotation_enabled:
            joint = random_rotation(joint, rotation_rng, theta=self.rotation_theta)
        streams = construct_shared_streams(joint) if self.return_streams else None
        selected = (
            streams[self.modality]
            if streams is not None
            else apply_modality(joint, self.modality)
        )
        padded_source_indices = np.full(128, -1, dtype=np.int64)
        padded_source_indices[: plan.source_frames_read] = plan.read_indices
        result: dict[str, Any] = {
            "skeleton": torch.from_numpy(selected),
            "label": torch.tensor(record.label, dtype=torch.long),
            "setup": torch.tensor(record.setup, dtype=torch.long),
            "sample_id": sample_id,
            "sequence_length": record.sequence_length,
            "indices": torch.from_numpy(np.array(plan.output_source_indices, copy=True)),
            "source_indices": torch.from_numpy(padded_source_indices),
            "source_index_count": plan.source_frames_read,
            "left_gather": torch.from_numpy(np.array(plan.left_gather, copy=True)),
            "right_gather": torch.from_numpy(np.array(plan.right_gather, copy=True)),
            "right_weight": torch.from_numpy(np.array(plan.right_weight, copy=True)),
            "temporal_mode": self.temporal_mode,
            "sampler": "bilinear_resize" if self.temporal_mode == "resize" else "uniform",
            "temporal_budget": self.num_frames,
            "crop_start": plan.crop_start,
            "cropped_sequence_length": plan.cropped_sequence_length,
            "source_frames_read": plan.source_frames_read,
            "source_bytes_logical": plan.source_frames_read * FRAME_BYTES,
            "selector_requires_full_sequence": False,
            "source_index_sha256": hashlib.sha256(
                plan.read_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }
        if streams is not None:
            result["streams"] = {
                stream: torch.from_numpy(value) for stream, value in streams.items()
            }
        return result


class SparseFourStreamDataset(SparseReferenceDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["return_streams"] = True
        kwargs.setdefault("modality", "joint")
        super().__init__(*args, **kwargs)
