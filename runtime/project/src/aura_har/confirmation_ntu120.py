from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aura_har.data.ntu import parse_ntu_name, validate_disjoint
from aura_har.data.ntu120 import NTU120_XSET_TRAIN_SETUPS, official_split_ntu120
from aura_har.utils.io import read_jsonl

NTU120_XSET_EXPECTED = {
    "input_files": 114_480,
    "ignored": 535,
    "usable": 113_945,
    "official_train": 54_468,
    "test": 59_477,
}
NTU120_XSET_EXPECTED_MANIFEST_HASHES = {
    "train": "fd037a9f71dbb00778bf71428892c8de407577402ee8330fdba0b02cea20edaa",
    "val": "2ece0ff18ea434cea3815dfeaf4832cb527d2f64dbbc5167f7389d9d2892ff35",
    "test": "5aed365468ad753ffd6887df88109841155f302db6cc033459c79cd9afd3194f",
}


def audit_ntu120_xset(
    root: str | Path,
    validation_setups: Sequence[int],
    full: bool = False,
) -> dict[str, Any]:
    """Audit official XSet and setup-disjoint validation without reading sample arrays."""
    dataset_root = Path(root).resolve()
    summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("dataset") != "ntu120" or summary.get("protocol") != "xset":
        raise ValueError("Expected a prepared NTU120 XSet dataset")
    manifests = {
        split: read_jsonl(dataset_root / "manifests" / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    validate_disjoint(manifests.values())
    expected_validation = {int(value) for value in validation_setups}
    if not expected_validation or not expected_validation.issubset(NTU120_XSET_TRAIN_SETUPS):
        raise ValueError("Validation setups must be non-empty and belong to even train setups")
    actual_validation = {int(row["setup"]) for row in manifests["val"]}
    if actual_validation != expected_validation:
        raise ValueError(
            "Validation setups differ from preregistration: "
            f"{sorted(actual_validation)} != {sorted(expected_validation)}"
        )
    train_setups = {int(row["setup"]) for row in manifests["train"]}
    test_setups = {int(row["setup"]) for row in manifests["test"]}
    if train_setups.intersection(actual_validation):
        raise ValueError("Train/validation setup overlap")
    if (train_setups | actual_validation).intersection(test_setups):
        raise ValueError("Official train/test setup overlap")

    for split, rows in manifests.items():
        for row in rows:
            meta = parse_ntu_name(str(row["sample_id"]))
            official = official_split_ntu120(meta, "xset")
            if split == "test" and official != "test":
                raise ValueError(f"Even-setup sample in test: {meta.sample_id}")
            if split in {"train", "val"} and official != "train":
                raise ValueError(f"Odd-setup sample outside test: {meta.sample_id}")
            if int(row["setup"]) != meta.setup or int(row["subject"]) != meta.subject:
                raise ValueError(f"Manifest metadata mismatch: {meta.sample_id}")
            if int(row["label"]) != meta.label:
                raise ValueError(f"Manifest label mismatch: {meta.sample_id}")
            if split == "val" and meta.setup not in expected_validation:
                raise ValueError(f"Unexpected validation setup: {meta.sample_id}")
            if split == "train" and meta.setup in expected_validation:
                raise ValueError(f"Validation setup leaked into train: {meta.sample_id}")

    counts = {split: len(rows) for split, rows in manifests.items()}
    if counts != summary.get("counts"):
        raise ValueError(f"Manifest/summary count mismatch: {counts} != {summary.get('counts')}")
    if full:
        expected = NTU120_XSET_EXPECTED
        if summary.get("input_files") != expected["input_files"]:
            raise ValueError("Unexpected raw input count")
        if summary.get("ignored") != expected["ignored"]:
            raise ValueError("Unexpected ignored-sample count")
        if sum(counts.values()) != expected["usable"]:
            raise ValueError("Unexpected usable sample count")
        if counts["test"] != expected["test"]:
            raise ValueError("Unexpected XSet test count")
        if counts["train"] + counts["val"] != expected["official_train"]:
            raise ValueError("Unexpected XSet train+validation count")
        if summary.get("manifest_hashes") != NTU120_XSET_EXPECTED_MANIFEST_HASHES:
            raise ValueError("Prepared manifests differ from the frozen XSet manifests")
    return {
        "status": "passed",
        "protocol": "ntu120_xset",
        "counts": counts,
        "validation_setups": sorted(actual_validation),
        "train_setups": sorted(train_setups),
        "test_setups": sorted(test_setups),
        "train_validation_setup_disjoint": True,
        "official_train_test_setup_disjoint": True,
        "sample_id_disjoint": True,
        "full_count_check": bool(full),
        "manifest_hashes": summary.get("manifest_hashes"),
        "frozen_manifest_hashes_match": (
            summary.get("manifest_hashes") == NTU120_XSET_EXPECTED_MANIFEST_HASHES
        ),
    }
