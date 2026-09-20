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
    official_split,
    parse_ntu_name,
    read_skeleton_file,
    validate_disjoint,
)
from aura_har.utils.io import (
    CANONICAL_JSONL_HASH_SCHEME,
    canonical_jsonl_sha256,
    write_json,
    write_jsonl,
)


def parse_subjects(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NTU skeleton manifests and NPZ samples")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["ntu60", "ntu120"], default="ntu60")
    parser.add_argument("--protocol", choices=["xsub", "xview", "xsetup"], default="xsub")
    parser.add_argument("--val-subjects", default="")
    parser.add_argument("--ignore-file", type=Path)
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
    val_subjects = parse_subjects(args.val_subjects)
    invalid_subjects = sorted(subject for subject in val_subjects if not 1 <= subject <= 40)
    if args.dataset == "ntu60" and invalid_subjects:
        raise ValueError(f"NTU60 validation subjects must be in 1..40: {invalid_subjects}")
    files = sorted(raw_dir.rglob("*.skeleton"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .skeleton files found under {raw_dir}")

    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    errors: list[dict[str, str]] = []
    skipped = 0
    for index, source in enumerate(files, start=1):
        try:
            meta = parse_ntu_name(source)
            if meta.sample_id in ignored:
                skipped += 1
                continue
            if args.dataset == "ntu60" and meta.action > 60:
                continue
            split = official_split(meta, args.dataset, args.protocol)
            if split == "train" and meta.subject in val_subjects:
                split = "val"
            target = sample_dir / f"{meta.sample_id}.npz"
            parser_meta: dict[str, int]
            if target.exists() and not args.overwrite:
                with np.load(target, allow_pickle=False) as archive:
                    skeleton = archive["skeleton"]
                    sequence_length = int(archive["sequence_length"])
                    parser_meta = {
                        "num_frames": int(skeleton.shape[1]),
                        "observed_body_ids": int(archive["observed_body_ids"])
                        if "observed_body_ids" in archive.files
                        else 0,
                        "retained_persons": int(archive["retained_persons"])
                        if "retained_persons" in archive.files
                        else 0,
                        "empty_frames": int(archive["empty_frames"])
                        if "empty_frames" in archive.files
                        else 0,
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
            row = {
                **meta.to_dict(),
                "label": meta.label,
                "split": split,
                "data_path": target.relative_to(output_dir).as_posix(),
                "raw_path": source.relative_to(raw_dir).as_posix(),
                "sequence_length": sequence_length,
                **parser_meta,
            }
            rows[split].append(row)
        except Exception as error:  # preserve failures in a machine-readable report
            errors.append({"path": str(source), "error": repr(error)})
            if args.strict:
                raise
        if index % 1000 == 0:
            print(f"Processed {index}/{len(files)} files")

    validate_disjoint(rows.values())
    if val_subjects and not rows["val"]:
        raise ValueError("Validation subjects were requested but the validation split is empty")
    actual_val_subjects = {int(row["subject"]) for row in rows["val"]}
    missing_val_subjects = sorted(val_subjects.difference(actual_val_subjects))
    if missing_val_subjects:
        raise ValueError(
            "Requested validation subjects are absent from the official training partition: "
            f"{missing_val_subjects}"
        )
    for split, split_rows in rows.items():
        write_jsonl(split_rows, manifest_dir / f"{split}.jsonl")
    distribution_path = output_dir / "class_distribution.csv"
    with distribution_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "label", "count"])
        for split, split_rows in rows.items():
            for label, count in sorted(Counter(row["label"] for row in split_rows).items()):
                writer.writerow([split, label, count])
    error_path = output_dir / "errors.json"
    write_json(errors, error_path)
    summary = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "raw_dir": str(raw_dir),
        "input_files": len(files),
        "ignored": skipped,
        "errors": len(errors),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "validation_subjects": sorted(val_subjects),
        "validation_rule": "held_out_subjects_within_official_training_partition",
        "manifest_hash_scheme": CANONICAL_JSONL_HASH_SCHEME,
        "manifest_hashes": {
            split: canonical_jsonl_sha256(manifest_dir / f"{split}.jsonl") for split in rows
        },
    }
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
