from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aura_har.data.ntu import official_split, parse_ntu_name, validate_disjoint
from aura_har.utils.io import read_jsonl, sha256_file

NTU60_XVIEW_EXPECTED = {
    "input_files": 56_880,
    "ignored": 302,
    "usable": 56_578,
    "official_train": 37_646,
    "test": 18_932,
}


def canonical_mapping_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_lock_sha256(lock_path: str | Path) -> str:
    return sha256_file(Path(lock_path))


def verify_protocol_lock(lock_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    path = Path(lock_path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    root = Path(project_root).resolve()
    if lock.get("schema") != "aura-har.confirmation-lock.v1":
        raise ValueError("Unsupported confirmation protocol lock schema")
    mismatches = []
    for relative, expected in lock.get("files", {}).items():
        target = root / relative
        actual = None if not target.is_file() else sha256_file(target)
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"Protocol lock mismatch: {mismatches[:3]}")
    return lock


def audit_ntu60_xview(
    root: str | Path,
    validation_subjects: Sequence[int],
    full: bool = False,
) -> dict[str, Any]:
    """Audit sample, camera and subject isolation for the pre-registered XView split."""
    dataset_root = Path(root).resolve()
    summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("dataset") != "ntu60" or summary.get("protocol") != "xview":
        raise ValueError("Expected a prepared NTU60 XView dataset")
    manifests = {
        split: read_jsonl(dataset_root / "manifests" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    validate_disjoint(manifests.values())
    expected_validation = {int(value) for value in validation_subjects}
    actual_validation = {int(row["subject"]) for row in manifests["val"]}
    if actual_validation != expected_validation:
        raise ValueError(
            f"Validation subjects differ from preregistration: "
            f"{sorted(actual_validation)} != {sorted(expected_validation)}"
        )
    train_subjects = {int(row["subject"]) for row in manifests["train"]}
    if train_subjects.intersection(actual_validation):
        raise ValueError("Train/validation subject overlap")

    for split, rows in manifests.items():
        for row in rows:
            meta = parse_ntu_name(str(row["sample_id"]))
            official = official_split(meta, "ntu60", "xview")
            if split == "test" and official != "test":
                raise ValueError(f"Non-camera-1 sample in test: {meta.sample_id}")
            if split in {"train", "val"} and official != "train":
                raise ValueError(f"Camera-1 sample outside test: {meta.sample_id}")
            if int(row["camera"]) != meta.camera or int(row["subject"]) != meta.subject:
                raise ValueError(f"Manifest metadata mismatch: {meta.sample_id}")
            if int(row["label"]) != meta.label:
                raise ValueError(f"Manifest label mismatch: {meta.sample_id}")
            if split == "val" and meta.subject not in expected_validation:
                raise ValueError(f"Unexpected validation subject: {meta.sample_id}")
            if split == "train" and meta.subject in expected_validation:
                raise ValueError(f"Validation subject leaked into train: {meta.sample_id}")

    counts = {split: len(rows) for split, rows in manifests.items()}
    if counts != summary.get("counts"):
        raise ValueError(f"Manifest/summary count mismatch: {counts} != {summary.get('counts')}")
    if full:
        expected = NTU60_XVIEW_EXPECTED
        if summary.get("input_files") != expected["input_files"]:
            raise ValueError("Unexpected raw input count")
        if summary.get("ignored") != expected["ignored"]:
            raise ValueError("Unexpected ignored-sample count")
        if sum(counts.values()) != expected["usable"]:
            raise ValueError("Unexpected usable sample count")
        if counts["test"] != expected["test"]:
            raise ValueError("Unexpected XView test count")
        if counts["train"] + counts["val"] != expected["official_train"]:
            raise ValueError("Unexpected XView train+validation count")
    return {
        "status": "passed",
        "protocol": "ntu60_xview",
        "counts": counts,
        "validation_subjects": sorted(actual_validation),
        "train_subjects": sorted(train_subjects),
        "train_validation_subject_disjoint": True,
        "train_validation_cameras": [2, 3],
        "test_cameras": [1],
        "sample_id_disjoint": True,
        "full_count_check": bool(full),
    }
