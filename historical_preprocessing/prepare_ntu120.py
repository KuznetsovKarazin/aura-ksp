#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from aura_har.data.ntu import (
    load_ignore_list,
    parse_ntu_name,
    read_skeleton_file,
    validate_disjoint,
)
from aura_har.data.ntu120 import (
    NTU120_XSET_TRAIN_SETUPS,
    NTU120_XSUB_TRAIN_SUBJECTS,
    official_split_ntu120,
)
from aura_har.utils.io import (
    CANONICAL_JSONL_HASH_SCHEME,
    canonical_jsonl_sha256,
    write_json,
    write_jsonl,
)


def parse_ids(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def validate_validation_partition(
    protocol: str, val_subjects: set[int], val_setups: set[int]
) -> None:
    if protocol == "xset":
        if val_subjects:
            raise ValueError("XSet validation must use --val-setups, not --val-subjects")
        if not val_setups:
            raise ValueError("XSet requires preregistered --val-setups")
        invalid = sorted(val_setups.difference(NTU120_XSET_TRAIN_SETUPS))
        if invalid:
            raise ValueError(f"XSet validation setups must be even IDs in 2..32: {invalid}")
    else:
        if val_setups:
            raise ValueError("XSub validation must use --val-subjects, not --val-setups")
        if not val_subjects:
            raise ValueError("XSub requires preregistered --val-subjects")
        invalid = sorted(val_subjects.difference(NTU120_XSUB_TRAIN_SUBJECTS))
        if invalid:
            raise ValueError(f"XSub validation subjects must belong to official train: {invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NTU120 skeleton manifests and NPZ")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["xsub", "xset"], default="xset")
    parser.add_argument("--val-subjects", default="")
    parser.add_argument("--val-setups", default="")
    parser.add_argument("--ignore-file", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sample_dir = output_dir / "samples"
    manifest_dir = output_dir / "manifests"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ignored = load_ignore_list(args.ignore_file)
    val_subjects = parse_ids(args.val_subjects)
    val_setups = parse_ids(args.val_setups)
    validate_validation_partition(args.protocol, val_subjects, val_setups)
    files = sorted(raw_dir.rglob("*.skeleton"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .skeleton files found under {raw_dir}")

    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    errors: list[dict[str, str]] = []
    skipped = 0
    seen: set[str] = set()
    for index, source in enumerate(files, start=1):
        try:
            meta = parse_ntu_name(source)
            if meta.sample_id in seen:
                raise ValueError(f"Duplicate NTU120 sample ID: {meta.sample_id}")
            seen.add(meta.sample_id)
            if meta.sample_id in ignored:
                skipped += 1
                continue
            split = official_split_ntu120(meta, args.protocol)
            if split == "train" and (
                (args.protocol == "xset" and meta.setup in val_setups)
                or (args.protocol == "xsub" and meta.subject in val_subjects)
            ):
                split = "val"
            target = sample_dir / f"{meta.sample_id}.npz"
            parser_meta: dict[str, int]
            if target.exists() and not args.overwrite:
                with np.load(target, allow_pickle=False) as archive:
                    skeleton = archive["skeleton"]
                    sequence_length = int(archive["sequence_length"])
                    parser_meta = {
                        "num_frames": int(skeleton.shape[1]),
                        "observed_body_ids": int(archive["observed_body_ids"]),
                        "retained_persons": int(archive["retained_persons"]),
                        "empty_frames": int(archive["empty_frames"]),
                    }
            else:
                skeleton, valid_frames, parser_meta = read_skeleton_file(source)
                valid_positions = np.flatnonzero(valid_frames)
                sequence_length = int(valid_positions[-1] + 1) if valid_positions.size else 1
                np.savez_compressed(
                    target,
                    skeleton=skeleton,
                    valid_frames=valid_frames,
                    sequence_length=np.int32(sequence_length),
                    observed_body_ids=np.int32(parser_meta["observed_body_ids"]),
                    retained_persons=np.int32(parser_meta["retained_persons"]),
                    empty_frames=np.int32(parser_meta["empty_frames"]),
                )
            rows[split].append(
                {
                    **meta.to_dict(),
                    "label": meta.label,
                    "split": split,
                    "data_path": target.relative_to(output_dir).as_posix(),
                    "raw_path": source.relative_to(raw_dir).as_posix(),
                    "sequence_length": sequence_length,
                    **parser_meta,
                }
            )
        except Exception as error:
            errors.append({"path": str(source), "error": repr(error)})
            if args.strict:
                raise
        if index % 1000 == 0:
            print(f"Processed {index}/{len(files)} files")

    validate_disjoint(rows.values())
    if not rows["val"]:
        raise ValueError("Preregistered validation partition is empty")
    for split, split_rows in rows.items():
        write_jsonl(split_rows, manifest_dir / f"{split}.jsonl")
    with (output_dir / "class_distribution.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "count"])
        for split, split_rows in rows.items():
            for label, count in sorted(Counter(row["label"] for row in split_rows).items()):
                writer.writerow([split, label, count])
    write_json(errors, output_dir / "errors.json")
    summary = {
        "dataset": "ntu120",
        "protocol": args.protocol,
        "raw_dir": str(raw_dir),
        "input_files": len(files),
        "ignored": skipped,
        "errors": len(errors),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "validation_subjects": sorted(val_subjects),
        "validation_setups": sorted(val_setups),
        "validation_rule": (
            "held_out_setups_within_official_training_partition"
            if args.protocol == "xset"
            else "held_out_subjects_within_official_training_partition"
        ),
        "manifest_hash_scheme": CANONICAL_JSONL_HASH_SCHEME,
        "manifest_hashes": {
            split: canonical_jsonl_sha256(manifest_dir / f"{split}.jsonl") for split in rows
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
