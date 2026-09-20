from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from aura_har.utils.io import (
    CANONICAL_JSONL_HASH_SCHEME,
    canonical_jsonl_sha256,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

STORE_SCHEMA = "aura-har.ntu120-frame-store.v1"
FRAME_LAYOUT = "T,C,V,M"
FRAME_SHAPE = (3, 25, 2)
FRAME_DTYPE = np.dtype("<f4")
FRAME_BYTES = int(np.prod(FRAME_SHAPE)) * FRAME_DTYPE.itemsize
DEFAULT_METADATA_NAME = "store.json"
DEFAULT_FRAMES_NAME = "frames.npy"
DEFAULT_INDEX_NAME = "index.jsonl"


@dataclass(frozen=True)
class SparseStoreRecord:
    sample_id: str
    label: int
    setup: int
    frame_offset: int
    sequence_length: int
    source_data_path: str
    content_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SparseStoreRecord:
        return cls(
            sample_id=str(value["sample_id"]),
            label=int(value["label"]),
            setup=int(value["setup"]),
            frame_offset=int(value["frame_offset"]),
            sequence_length=int(value["sequence_length"]),
            source_data_path=str(value["source_data_path"]),
            content_sha256=str(value["content_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_metadata_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    return path / DEFAULT_METADATA_NAME if path.is_dir() else path


def _portable_relative_path(value: str | Path) -> Path:
    normalized = str(value).replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"Store source path must be portable and relative: {value}")
    return Path(*pure.parts)


def _resolve_source_path(root: Path, relative: str | Path) -> Path:
    candidate = (root / _portable_relative_path(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Store source path escapes its root: {relative}") from error
    return candidate


def _semantic_frames_sha256(frames_tcvm: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(frames_tcvm, dtype=FRAME_DTYPE)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _load_source_frames(path: Path, expected_length: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "skeleton" not in archive or "sequence_length" not in archive:
            raise ValueError(f"Processed sample lacks skeleton/sequence_length: {path}")
        sequence = np.asarray(archive["skeleton"])
        archive_length = int(archive["sequence_length"])
    if sequence.ndim != 4 or tuple(sequence.shape[0:1] + sequence.shape[2:]) != FRAME_SHAPE:
        raise ValueError(f"Expected C,T,V,M={FRAME_SHAPE[0]},T,25,2, got {sequence.shape}: {path}")
    if archive_length != expected_length:
        raise ValueError(
            f"Sequence length mismatch for {path}: archive={archive_length}, "
            f"manifest={expected_length}"
        )
    if expected_length <= 0 or expected_length > sequence.shape[1]:
        raise ValueError(
            f"Invalid sequence length for {path}: {expected_length} not in [1,{sequence.shape[1]}]"
        )
    frames = sequence[:, :expected_length].transpose(1, 0, 2, 3)
    frames = np.ascontiguousarray(frames, dtype=FRAME_DTYPE)
    if not np.isfinite(frames).all():
        raise ValueError(f"Non-finite skeleton coordinates in {path}")
    return frames


def _validate_manifest_rows(
    rows: list[dict[str, Any]], expected_samples: int, expected_setups: Collection[int]
) -> tuple[list[dict[str, Any]], tuple[int, ...], int]:
    if len(rows) != int(expected_samples):
        raise ValueError(f"Unexpected train sample count: {len(rows)} != {expected_samples}")
    required = {"sample_id", "label", "setup", "data_path", "sequence_length"}
    sample_ids: list[str] = []
    total_frames = 0
    for row in rows:
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"Manifest row lacks required fields {missing}: {row}")
        sample_id = str(row["sample_id"])
        if not sample_id:
            raise ValueError("Manifest contains an empty sample_id")
        sample_ids.append(sample_id)
        length = int(row["sequence_length"])
        if length <= 0:
            raise ValueError(f"Non-positive sequence_length for {sample_id}: {length}")
        total_frames += length
        _portable_relative_path(str(row["data_path"]))
        split = row.get("split")
        if split is not None and str(split).lower() != "train":
            raise ValueError(f"Sparse train store refuses split={split!r}: {sample_id}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Train manifest contains duplicate sample IDs")
    actual_setups = tuple(sorted({int(row["setup"]) for row in rows}))
    frozen_setups = tuple(sorted(int(value) for value in expected_setups))
    if actual_setups != frozen_setups:
        raise ValueError(f"Unexpected train setups: {actual_setups} != {frozen_setups}")
    return rows, actual_setups, total_frames


class NTU120FrameStore:
    """Read-only source-indexed frame store backed by one portable ``.npy`` memmap."""

    def __init__(self, metadata_path: str | Path) -> None:
        self.metadata_path = _resolve_metadata_path(metadata_path)
        self.root = self.metadata_path.parent
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema") != STORE_SCHEMA:
            raise ValueError(f"Unsupported sparse-store schema: {self.metadata.get('schema')}")
        if self.metadata.get("frame_layout") != FRAME_LAYOUT:
            raise ValueError("Sparse-store frame layout mismatch")
        if tuple(self.metadata.get("frame_shape_cvm", ())) != FRAME_SHAPE:
            raise ValueError("Sparse-store frame shape mismatch")
        if np.dtype(self.metadata.get("dtype")) != FRAME_DTYPE:
            raise ValueError("Sparse-store dtype mismatch")

        self.frames_path = self.root / _portable_relative_path(self.metadata["frames_path"])
        self.index_path = self.root / _portable_relative_path(self.metadata["index_path"])
        if sha256_file(self.index_path) != str(self.metadata["index_sha256"]):
            raise ValueError("Sparse-store index SHA-256 mismatch")
        rows = read_jsonl(self.index_path)
        records = [SparseStoreRecord.from_dict(row) for row in rows]
        self._records = self._validate_records(records)
        self._frames: np.ndarray | None = None
        self._read_calls = 0
        self._validate_array_header()

    def _validate_records(
        self, records: list[SparseStoreRecord]
    ) -> dict[str, SparseStoreRecord]:
        if len(records) != int(self.metadata["samples"]):
            raise ValueError("Sparse-store index sample count mismatch")
        expected_offset = 0
        mapped: dict[str, SparseStoreRecord] = {}
        for record in records:
            if record.sample_id in mapped:
                raise ValueError(f"Duplicate sparse-store sample ID: {record.sample_id}")
            if record.sequence_length <= 0:
                raise ValueError(f"Non-positive store length: {record.sample_id}")
            if record.frame_offset != expected_offset:
                raise ValueError(
                    f"Non-contiguous frame offset for {record.sample_id}: "
                    f"{record.frame_offset} != {expected_offset}"
                )
            if len(record.content_sha256) != 64:
                raise ValueError(f"Invalid semantic SHA-256 for {record.sample_id}")
            _portable_relative_path(record.source_data_path)
            mapped[record.sample_id] = record
            expected_offset += record.sequence_length
        if expected_offset != int(self.metadata["total_frames"]):
            raise ValueError("Sparse-store total frame count mismatch")
        actual_setups = sorted({record.setup for record in records})
        if actual_setups != [int(value) for value in self.metadata["setups"]]:
            raise ValueError("Sparse-store setup inventory mismatch")
        return mapped

    def _validate_array_header(self) -> None:
        frames = np.load(self.frames_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (int(self.metadata["total_frames"]), *FRAME_SHAPE)
        if frames.shape != expected_shape:
            raise ValueError(f"Sparse-store array shape mismatch: {frames.shape} != {expected_shape}")
        if frames.dtype != FRAME_DTYPE:
            raise ValueError(f"Sparse-store array dtype mismatch: {frames.dtype} != {FRAME_DTYPE}")
        if not frames.flags.c_contiguous:
            raise ValueError("Sparse-store array must be C-contiguous")

    @property
    def frames(self) -> np.ndarray:
        if self._frames is None:
            self._frames = np.load(self.frames_path, mmap_mode="r", allow_pickle=False)
        return self._frames

    @property
    def read_calls(self) -> int:
        return self._read_calls

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, sample_id: object) -> bool:
        return str(sample_id) in self._records

    def record(self, sample_id: str) -> SparseStoreRecord:
        try:
            return self._records[str(sample_id)]
        except KeyError as error:
            raise KeyError(f"Sample is absent from sparse store: {sample_id}") from error

    def read_frames(self, sample_id: str, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Read selected source frames once and return contiguous ``C,K,V,M`` float32."""
        record = self.record(sample_id)
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("Sparse source indices must be a non-empty one-dimensional array")
        if np.any(selected < 0) or np.any(selected >= record.sequence_length):
            raise IndexError(
                f"Sparse source index outside [0,{record.sequence_length}): {sample_id}"
            )
        self._read_calls += 1
        absolute = selected + record.frame_offset
        frames = np.asarray(self.frames[absolute], dtype=FRAME_DTYPE)
        return np.ascontiguousarray(frames.transpose(1, 0, 2, 3), dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_frames"] = None
        state["_read_calls"] = 0
        return state


def build_ntu120_train_store(
    manifest: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    expected_manifest_sha256: str,
    expected_samples: int,
    expected_setups: Collection[int],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize the audited NTU120 train partition as frame-major float32 storage."""
    manifest_path = Path(manifest).expanduser().resolve()
    source_path = Path(source_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    actual_manifest_sha256 = canonical_jsonl_sha256(manifest_path)
    if actual_manifest_sha256 != str(expected_manifest_sha256):
        raise ValueError(
            "Canonical train manifest SHA-256 mismatch: "
            f"{actual_manifest_sha256} != {expected_manifest_sha256}"
        )
    rows, setups, total_frames = _validate_manifest_rows(
        read_jsonl(manifest_path), expected_samples, expected_setups
    )

    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / DEFAULT_METADATA_NAME
    frames_path = output_path / DEFAULT_FRAMES_NAME
    index_path = output_path / DEFAULT_INDEX_NAME
    partial_frames = output_path / f".{DEFAULT_FRAMES_NAME}.partial"
    partial_index = output_path / f".{DEFAULT_INDEX_NAME}.partial"
    partial_metadata = output_path / f".{DEFAULT_METADATA_NAME}.partial"
    final_targets = (metadata_path, frames_path, index_path)
    if any(path.exists() for path in final_targets) and not overwrite:
        existing = [str(path) for path in final_targets if path.exists()]
        raise FileExistsError(f"Sparse-store artifacts already exist: {existing}")
    for path in (partial_frames, partial_index, partial_metadata):
        if path.exists():
            path.unlink()

    records: list[SparseStoreRecord] = []
    try:
        mapped = np.lib.format.open_memmap(
            partial_frames,
            mode="w+",
            dtype=FRAME_DTYPE,
            shape=(total_frames, *FRAME_SHAPE),
        )
        offset = 0
        for row in rows:
            sample_id = str(row["sample_id"])
            length = int(row["sequence_length"])
            relative = str(row["data_path"]).replace("\\", "/")
            source = _resolve_source_path(source_path, relative)
            frames = _load_source_frames(source, length)
            mapped[offset : offset + length] = frames
            records.append(
                SparseStoreRecord(
                    sample_id=sample_id,
                    label=int(row["label"]),
                    setup=int(row["setup"]),
                    frame_offset=offset,
                    sequence_length=length,
                    source_data_path=relative,
                    content_sha256=_semantic_frames_sha256(frames),
                )
            )
            offset += length
        mapped.flush()
        del mapped
        if offset != total_frames:
            raise AssertionError(f"Sparse-store write length mismatch: {offset} != {total_frames}")
        write_jsonl((record.to_dict() for record in records), partial_index)
        frames_sha256 = sha256_file(partial_frames)
        index_sha256 = sha256_file(partial_index)
        metadata: dict[str, Any] = {
            "schema": STORE_SCHEMA,
            "source_manifest_sha256": actual_manifest_sha256,
            "samples": len(records),
            "setups": list(setups),
            "total_frames": total_frames,
            "frame_layout": FRAME_LAYOUT,
            "frame_shape_cvm": list(FRAME_SHAPE),
            "dtype": FRAME_DTYPE.str,
            "frame_bytes": FRAME_BYTES,
            "logical_data_bytes": total_frames * FRAME_BYTES,
            "frames_path": DEFAULT_FRAMES_NAME,
            "frames_sha256": frames_sha256,
            "index_path": DEFAULT_INDEX_NAME,
            "index_sha256": index_sha256,
            "source_kind": "processed_ntu_npz_train_only",
            "test_read": False,
        }
        write_json(metadata, partial_metadata)
        os.replace(partial_frames, frames_path)
        os.replace(partial_index, index_path)
        os.replace(partial_metadata, metadata_path)
    finally:
        for path in (partial_frames, partial_index, partial_metadata):
            if path.exists():
                path.unlink()

    report = audit_ntu120_train_store(
        metadata_path,
        source_manifest=manifest_path,
        source_root=source_path,
        deep=True,
    )
    report["metadata_path"] = str(metadata_path)
    return report


def audit_ntu120_train_store(
    metadata_path: str | Path,
    source_manifest: str | Path | None = None,
    source_root: str | Path | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """Verify structure/hashes and optionally every stored frame against source train NPZ."""
    store = NTU120FrameStore(metadata_path)
    metadata = store.metadata
    checks: dict[str, Any] = {
        "schema": True,
        "index_sha256": True,
        "array_header": True,
        "offsets_contiguous": True,
        "sample_ids_unique": True,
    }
    if source_root is not None and source_manifest is None:
        raise ValueError("source_root requires source_manifest for a bounded train-only audit")

    manifest_rows: list[dict[str, Any]] | None = None
    if source_manifest is not None:
        manifest_path = Path(source_manifest).expanduser().resolve()
        manifest_sha = canonical_jsonl_sha256(manifest_path)
        if manifest_sha != str(metadata["source_manifest_sha256"]):
            raise ValueError("Audit source manifest differs from the stored train manifest")
        manifest_rows = read_jsonl(manifest_path)
        if len(manifest_rows) != len(store):
            raise ValueError("Audit source manifest sample count mismatch")
        manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
        if manifest_ids != list(store._records):
            raise ValueError("Audit source manifest ordering differs from sparse index")
        for row in manifest_rows:
            record = store.record(str(row["sample_id"]))
            if (
                int(row["label"]) != record.label
                or int(row["setup"]) != record.setup
                or int(row["sequence_length"]) != record.sequence_length
            ):
                raise ValueError(f"Audit source metadata mismatch: {record.sample_id}")
        checks["source_manifest_sha256"] = True
        checks["source_manifest_metadata"] = True

    if deep:
        if sha256_file(store.frames_path) != str(metadata["frames_sha256"]):
            raise ValueError("Sparse-store frame-array SHA-256 mismatch")
        checks["frames_sha256"] = True
        for record in store._records.values():
            start = record.frame_offset
            stop = start + record.sequence_length
            frames = np.asarray(store.frames[start:stop], dtype=FRAME_DTYPE)
            if not np.isfinite(frames).all():
                raise ValueError(f"Non-finite stored frames: {record.sample_id}")
            if _semantic_frames_sha256(frames) != record.content_sha256:
                raise ValueError(f"Stored semantic frame hash mismatch: {record.sample_id}")
        checks["semantic_content_sha256"] = True

        if source_root is not None and manifest_rows is not None:
            root = Path(source_root).expanduser().resolve()
            for row in manifest_rows:
                record = store.record(str(row["sample_id"]))
                source = _resolve_source_path(root, str(row["data_path"]))
                source_frames = _load_source_frames(source, record.sequence_length)
                start = record.frame_offset
                stop = start + record.sequence_length
                stored_frames = np.asarray(store.frames[start:stop], dtype=FRAME_DTYPE)
                if not np.array_equal(source_frames, stored_frames):
                    raise ValueError(f"Sparse/source frame mismatch: {record.sample_id}")
            checks["source_frame_equality"] = True

    return {
        "schema": "aura-har.ntu120-frame-store-audit.v1",
        "status": "passed",
        "store_schema": metadata["schema"],
        "metadata_path": str(store.metadata_path),
        "manifest_hash_scheme": CANONICAL_JSONL_HASH_SCHEME,
        "source_manifest_sha256": str(metadata["source_manifest_sha256"]),
        "store_metadata_sha256": sha256_file(store.metadata_path),
        "store_index_sha256": str(metadata["index_sha256"]),
        "store_frames_sha256": str(metadata["frames_sha256"]),
        "samples": len(store),
        "setups": list(metadata["setups"]),
        "total_frames": int(metadata["total_frames"]),
        "logical_data_bytes": int(metadata["logical_data_bytes"]),
        "deep": bool(deep),
        "checks": checks,
        "test_read": False,
    }
