from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aura_har.sampling import build_sampler
from aura_har.utils.io import read_jsonl

from .reference import apply_modality, exact_temporal_sample, random_rotation, valid_crop_resize


def portable_manifest_path(value: str | Path) -> Path:
    """Interpret legacy Windows manifest paths on every supported OS."""
    return Path(str(value).replace("\\", "/"))


def normalize_skeleton(sequence: np.ndarray) -> np.ndarray:
    """Center on the primary root while preserving inter-person displacement."""
    output = sequence.astype(np.float32, copy=True)
    _, time, joints, persons = output.shape
    if time == 0:
        return output
    primary = output[:, :, 0, 0]
    valid_primary = np.any(np.abs(primary) > 0, axis=0)
    for frame in range(time):
        if valid_primary[frame]:
            for person in range(persons):
                valid_joints = np.any(np.abs(output[:, frame, :, person]) > 0, axis=0)
                coords = output[:, frame, :, person]
                coords[:, valid_joints] -= primary[:, frame, None]
                output[:, frame, :, person] = coords
    if joints > 8:
        shoulders = np.linalg.norm(output[:, :, 4, 0] - output[:, :, 8, 0], axis=0)
        scale_values = shoulders[shoulders > 1e-6]
    else:
        scale_values = np.asarray([], dtype=np.float32)
    scale = float(np.median(scale_values)) if scale_values.size else float(output.std())
    if scale > 1e-6:
        output /= scale
    return output


class SkeletonDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        sampler_name: str,
        num_frames: int,
        train: bool,
        seed: int,
        normalize: bool = True,
        train_random_offset: bool = True,
    ) -> None:
        self.rows = read_jsonl(manifest)
        self.root = Path(root)
        self.num_frames = int(num_frames)
        self.train = train
        self.seed = int(seed)
        self.epoch = 0
        self.normalize = normalize
        self.sampler = build_sampler(
            sampler_name, train=train, random_offset=train and train_random_offset
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        archive_path = self.root / portable_manifest_path(row["data_path"])
        with np.load(archive_path, allow_pickle=False) as archive:
            sequence = archive["skeleton"].astype(np.float32)
            length = int(archive["sequence_length"])
        if self.normalize:
            sequence = normalize_skeleton(sequence)
        stable = zlib.crc32(str(row["sample_id"]).encode("utf-8"))
        rng_seed = self.seed + stable + (self.epoch * 1_000_003 if self.train else 0)
        rng = np.random.default_rng(rng_seed)
        indices = self.sampler(sequence, length, self.num_frames, rng)
        sampled = np.ascontiguousarray(sequence[:, indices])
        return {
            "skeleton": torch.from_numpy(sampled),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "sequence_length": length,
            "indices": torch.from_numpy(indices),
        }


def _read_manifests(manifest: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    sources = [manifest] if isinstance(manifest, (str, Path)) else list(manifest)
    if not sources:
        raise ValueError("At least one manifest is required")
    rows = [row for source in sources for row in read_jsonl(source)]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Combined manifests contain duplicate sample IDs")
    return rows


class ReferenceSkeletonDataset(Dataset[dict[str, Any]]):
    """Per-sample NTU feeder with the official CTR-GCN preprocessing order."""

    def __init__(
        self,
        manifest: str | Path | Sequence[str | Path],
        root: str | Path,
        num_frames: int,
        train: bool,
        seed: int,
        modality: str,
        p_interval_train: Sequence[float] = (0.5, 1.0),
        p_interval_eval: Sequence[float] = (0.95,),
        random_rotation_enabled: bool = True,
        rotation_theta: float = 0.3,
        temporal_mode: str = "resize",
        sampler_name: str = "uniform",
        train_random_offset: bool = True,
    ) -> None:
        self.rows = _read_manifests(manifest)
        self.root = Path(root)
        self.num_frames = int(num_frames)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.modality = str(modality).lower()
        self.p_interval_train = tuple(float(value) for value in p_interval_train)
        self.p_interval_eval = tuple(float(value) for value in p_interval_eval)
        self.random_rotation_enabled = bool(random_rotation_enabled)
        self.rotation_theta = float(rotation_theta)
        self.temporal_mode = str(temporal_mode).lower()
        if self.temporal_mode not in {"resize", "exact_k"}:
            raise ValueError(f"Unknown reference temporal_mode: {temporal_mode}")
        self.sampler_name = str(sampler_name).lower()
        self.train_random_offset = bool(train_random_offset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        archive_path = self.root / portable_manifest_path(row["data_path"])
        with np.load(archive_path, allow_pickle=False) as archive:
            sequence = archive["skeleton"].astype(np.float32)
            length = int(archive["sequence_length"])
        stable = zlib.crc32(str(row["sample_id"]).encode("utf-8"))
        rng_seed = self.seed + stable + (self.epoch * 1_000_003 if self.train else 0)
        rng = np.random.default_rng(rng_seed)
        interval = self.p_interval_train if self.train else self.p_interval_eval
        metadata: dict[str, int | bool | str]
        if self.temporal_mode == "resize":
            sequence, indices = valid_crop_resize(
                sequence, length, interval, self.num_frames, rng
            )
            if self.train and self.random_rotation_enabled:
                sequence = random_rotation(sequence, rng, theta=self.rotation_theta)
            sequence = apply_modality(sequence, self.modality)
            metadata = {
                "temporal_mode": "resize",
                "sampler": "bilinear_resize",
                "temporal_budget": self.num_frames,
                "crop_start": int(indices.min()),
                "cropped_sequence_length": int(indices.max() - indices.min() + 1),
                "source_frames_observed": int(indices.max() - indices.min() + 1),
                "selector_requires_full_sequence": True,
            }
        else:
            cropped, indices, metadata = exact_temporal_sample(
                sequence,
                length,
                interval,
                self.num_frames,
                self.sampler_name,
                rng,
                train=self.train,
                train_random_offset=self.train_random_offset,
            )
            local_indices = indices - int(metadata["crop_start"])
            if np.any(local_indices < 0) or np.any(local_indices >= cropped.shape[1]):
                raise AssertionError("Exact-K global/local index conversion failed")
            sequence = np.ascontiguousarray(cropped[:, local_indices], dtype=np.float32)
            if self.train and self.random_rotation_enabled:
                sequence = random_rotation(sequence, rng, theta=self.rotation_theta)
            sequence = apply_modality(sequence, self.modality)
        result: dict[str, Any] = {
            "skeleton": torch.from_numpy(sequence),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "sequence_length": length,
            "indices": torch.from_numpy(indices),
        }
        result.update(metadata)
        return result


def build_dataset(
    config: dict[str, Any], split: str, seed: int
) -> SkeletonDataset | ReferenceSkeletonDataset:
    data = config["data"]
    manifest = data[f"{split}_manifest"]
    if str(data.get("preprocessing", "historical")).lower() == "ctrgcn_reference":
        return ReferenceSkeletonDataset(
            manifest=manifest,
            root=data["root"],
            num_frames=int(data["num_frames"]),
            train=split == "train",
            seed=seed,
            modality=str(data.get("modality", "joint")),
            p_interval_train=data.get("p_interval_train", [0.5, 1.0]),
            p_interval_eval=data.get("p_interval_eval", [0.95]),
            random_rotation_enabled=bool(data.get("random_rotation", True)),
            rotation_theta=float(data.get("rotation_theta", 0.3)),
            temporal_mode=str(data.get("temporal_mode", "resize")),
            sampler_name=str(data.get("sampler", "uniform")),
            train_random_offset=bool(data.get("train_random_offset", True)),
        )
    return SkeletonDataset(
        manifest=manifest,
        root=data["root"],
        sampler_name=data.get("sampler", "uniform"),
        num_frames=int(data["num_frames"]),
        train=split == "train",
        seed=seed,
        normalize=bool(data.get("normalize", True)),
        train_random_offset=bool(data.get("train_random_offset", True)),
    )
