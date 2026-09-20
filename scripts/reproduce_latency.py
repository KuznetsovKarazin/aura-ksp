#!/usr/bin/env python3
"""Validate and analyze paired AURA-KSP three-stream GPU latency traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "aura-ksp-stream3-latency-microbenchmark-v1"
SOURCE_PROTOCOL_ID = "aura-ksp-ntu120-xset-trainonly-sparse-v1"
SCHEMA = "aura-har.ksp-stream3-gpu-latency-trace.v1"
SYSTEMS = ("full4_k64", "stream3_drop_bone_motion")
REGIMES = ("warm_reuse", "fresh_mapping")
FOLDS = (1, 2, 3)
REPLICATES = (1, 2, 3)
SETUPS_BY_FOLD = {1: (4, 12, 18, 26), 2: (6, 14, 22, 28), 3: (8, 16, 24, 32)}
FIELDS = ("latency_ms", "acquisition_ms", "h2d_ms", "inference_fusion_ms")
SAMPLES_PER_SETUP = 100
WARMUP = 30
BOOTSTRAP_SEED = 314_159
BOOTSTRAP_RESAMPLES = 10_000
DRAW_SHA256 = "1147c97e41e81e04b00f62aa3de02e282ff292f57b179668e797d53f1e831212"
MIN_SAVING = 0.20
FAMILY_ALPHA = 0.05


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(archive: Any, name: str) -> Any:
    value = np.asarray(archive[name])
    return value.item() if value.shape == () else value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def basic_lcb(estimate: float, distribution: np.ndarray, alpha: float) -> float:
    centered = np.asarray(distribution, dtype=np.float64) - float(estimate)
    return float(estimate - np.quantile(centered, 1 - alpha, method="higher"))


def margin_p_value(estimate: float, distribution: np.ndarray, margin: float) -> float:
    centered = np.asarray(distribution, dtype=np.float64) - float(estimate)
    return float((1 + np.sum(centered >= estimate - margin)) / (len(centered) + 1))


def paired_saving(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_total = float(np.sum(reference))
    if reference_total <= 0:
        raise ValueError("Reference latency total must be positive")
    return float(1.0 - float(np.sum(candidate)) / reference_total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    return parser.parse_args()


def load_traces(root: Path, protocol_sha256: str) -> dict[str, Any]:
    paths = sorted(root.rglob("*.npz"))
    if len(paths) != 18:
        raise RuntimeError(f"Expected 18 complete traces, found {len(paths)}")
    observations: dict[str, dict[str, dict[str, list[float]]]] = {
        regime: {field: defaultdict(list) for field in FIELDS} for regime in REGIMES
    }
    sample_setup: dict[str, int] = {}
    sample_fold: dict[str, int] = {}
    seen_jobs: set[tuple[int, str, int]] = set()
    environment_fingerprint: tuple[str, str, str] | None = None
    fold_provenance: dict[int, tuple[str, str]] = {}
    fold_samples: dict[int, set[str]] = {}
    trace_hashes: list[dict[str, Any]] = []
    authorization_sha256: str | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "schema", "protocol_id", "source_protocol_id", "base_seed", "variant",
                "fold", "cache_regime", "replicate", "systems", "system_streams_json",
                "sample_ids", "setups", *FIELDS, "source_frames_read", "source_bytes_logical",
                "execution_order", "full_prior", "three_stream_prior", "samples_per_setup",
                "warmup", "process_repetitions", "batch_size", "device", "device_name",
                "environment_json", "checkpoint_sha256_json", "config_sha256_json",
                "authorization_sha256", "protocol_sha256", "old_validation_used", "test_read",
                "training_performed", "external_confirmation",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"Trace lacks fields {sorted(missing)}: {path}")
            if str(scalar(archive, "schema")) != SCHEMA:
                raise RuntimeError(f"Trace schema mismatch: {path}")
            if str(scalar(archive, "protocol_id")) != PROTOCOL_ID:
                raise RuntimeError(f"Protocol mismatch: {path}")
            if str(scalar(archive, "source_protocol_id")) != SOURCE_PROTOCOL_ID:
                raise RuntimeError(f"Source protocol mismatch: {path}")
            if int(scalar(archive, "base_seed")) != 271_828 or str(scalar(archive, "variant")) != "k64_resize_reference":
                raise RuntimeError(f"Frozen model identity mismatch: {path}")
            if str(scalar(archive, "protocol_sha256")) != protocol_sha256:
                raise RuntimeError(f"Protocol hash mismatch: {path}")
            if any(bool(scalar(archive, name)) for name in (
                "old_validation_used", "test_read", "training_performed", "external_confirmation"
            )):
                raise RuntimeError(f"Forbidden provenance flag: {path}")
            if np.asarray(archive["systems"]).astype(str).tolist() != list(SYSTEMS):
                raise RuntimeError(f"System order mismatch: {path}")
            if json.loads(str(scalar(archive, "system_streams_json"))) != {
                "full4_k64": list(("joint", "bone", "joint_motion", "bone_motion")),
                "stream3_drop_bone_motion": list(("joint", "bone", "joint_motion")),
            }:
                raise RuntimeError(f"System stream boundary mismatch: {path}")
            if not np.array_equal(np.asarray(archive["full_prior"], dtype=np.float32), np.asarray((0.3, 0.3, 0.2, 0.2), dtype=np.float32)):
                raise RuntimeError(f"Full prior mismatch: {path}")
            expected_three = np.asarray((0.375, 0.375, 0.25), dtype=np.float32)
            if not np.array_equal(np.asarray(archive["three_stream_prior"], dtype=np.float32), expected_three):
                raise RuntimeError(f"Three-stream prior mismatch: {path}")
            frozen = {"samples_per_setup": 100, "warmup": 30, "process_repetitions": 3, "batch_size": 1}
            for name, expected in frozen.items():
                if int(scalar(archive, name)) != expected:
                    raise RuntimeError(f"Frozen scalar mismatch for {name}: {path}")
            fold = int(scalar(archive, "fold"))
            regime = str(scalar(archive, "cache_regime"))
            replicate = int(scalar(archive, "replicate"))
            job = (fold, regime, replicate)
            if fold not in FOLDS or regime not in REGIMES or replicate not in REPLICATES or job in seen_jobs:
                raise RuntimeError(f"Duplicate or unknown job {job}")
            seen_jobs.add(job)
            sample_ids = np.asarray(archive["sample_ids"]).astype(str)
            setups = np.asarray(archive["setups"], dtype=np.int64)
            if len(sample_ids) != 400 or len(set(sample_ids.tolist())) != 400:
                raise RuntimeError(f"Trace must contain 400 unique samples: {path}")
            if Counter(setups.tolist()) != Counter({setup: 100 for setup in SETUPS_BY_FOLD[fold]}):
                raise RuntimeError(f"Setup balance mismatch: {path}")
            sample_set = set(sample_ids.tolist())
            if fold in fold_samples and fold_samples[fold] != sample_set:
                raise RuntimeError(f"Sample selection drift in fold {fold}")
            fold_samples[fold] = sample_set
            measurements = {field: np.asarray(archive[field], dtype=np.float64) for field in FIELDS}
            for field, values in measurements.items():
                if values.shape != (400, 2) or np.any(~np.isfinite(values)) or np.any(values <= 0):
                    raise RuntimeError(f"Invalid {field} values: {path}")
            if np.any(measurements["latency_ms"] + 1e-9 < (
                measurements["acquisition_ms"] + measurements["h2d_ms"] + measurements["inference_fusion_ms"]
            )):
                raise RuntimeError(f"Stage timings exceed total latency: {path}")
            frames = np.asarray(archive["source_frames_read"], dtype=np.int64)
            logical_bytes = np.asarray(archive["source_bytes_logical"], dtype=np.int64)
            if frames.shape != (400, 2) or logical_bytes.shape != (400, 2):
                raise RuntimeError(f"Acquisition counter shape mismatch: {path}")
            if not np.array_equal(frames[:, 0], frames[:, 1]) or not np.array_equal(logical_bytes, frames * 600):
                raise RuntimeError(f"Paired acquisition boundary mismatch: {path}")
            order = np.asarray(archive["execution_order"], dtype=np.int64)
            if order.shape != (400, 2) or any(sorted(row.tolist()) != [0, 1] for row in order):
                raise RuntimeError(f"Invalid paired execution order: {path}")
            fingerprint = (
                str(scalar(archive, "device")), str(scalar(archive, "device_name")),
                str(scalar(archive, "environment_json")),
            )
            if not fingerprint[0].startswith("cuda"):
                raise RuntimeError(f"Non-CUDA trace: {path}")
            if environment_fingerprint is None:
                environment_fingerprint = fingerprint
            elif fingerprint != environment_fingerprint:
                raise RuntimeError("Environment drift across traces")
            provenance = (
                str(scalar(archive, "checkpoint_sha256_json")),
                str(scalar(archive, "config_sha256_json")),
            )
            if fold in fold_provenance and fold_provenance[fold] != provenance:
                raise RuntimeError(f"Checkpoint/config drift in fold {fold}")
            fold_provenance[fold] = provenance
            current_auth = str(scalar(archive, "authorization_sha256"))
            if authorization_sha256 is None:
                authorization_sha256 = current_auth
            elif current_auth != authorization_sha256:
                raise RuntimeError("Authorization drift across traces")
            for row, sample_id in enumerate(sample_ids):
                setup = int(setups[row])
                if sample_id in sample_setup and sample_setup[sample_id] != setup:
                    raise RuntimeError(f"Setup mismatch for {sample_id}")
                if sample_id in sample_fold and sample_fold[sample_id] != fold:
                    raise RuntimeError(f"Fold mismatch for {sample_id}")
                sample_setup[sample_id] = setup
                sample_fold[sample_id] = fold
                for column, system in enumerate(SYSTEMS):
                    key = f"{sample_id}\0{system}"
                    for field, values in measurements.items():
                        observations[regime][field][key].append(float(values[row, column]))
        trace_hashes.append({"path": str(path), "sha256": sha256_file(path), "fold": fold, "cache_regime": regime, "replicate": replicate})
    expected_jobs = {(fold, regime, replicate) for fold in FOLDS for regime in REGIMES for replicate in REPLICATES}
    if seen_jobs != expected_jobs:
        raise RuntimeError("Incomplete job matrix")
    if len(sample_setup) != 1200:
        raise RuntimeError(f"Expected 1200 unique selected samples, found {len(sample_setup)}")
    medians: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    sample_ids = np.asarray(sorted(sample_setup))
    for regime in REGIMES:
        medians[regime] = {}
        for system in SYSTEMS:
            medians[regime][system] = {}
            for field in FIELDS:
                values = []
                for sample_id in sample_ids:
                    repetitions = observations[regime][field].get(f"{sample_id}\0{system}", [])
                    if len(repetitions) != 3:
                        raise RuntimeError(f"Expected three repetitions for {regime}/{sample_id}/{system}/{field}")
                    values.append(float(np.median(repetitions)))
                medians[regime][system][field] = np.asarray(values, dtype=np.float64)
    return {
        "sample_ids": sample_ids,
        "setups": np.asarray([sample_setup[value] for value in sample_ids], dtype=np.int64),
        "folds": np.asarray([sample_fold[value] for value in sample_ids], dtype=np.int64),
        "medians": medians,
        "trace_hashes": trace_hashes,
        "device": environment_fingerprint[0] if environment_fingerprint else None,
        "device_name": environment_fingerprint[1] if environment_fingerprint else None,
        "environment_json": environment_fingerprint[2] if environment_fingerprint else None,
        "authorization_sha256": authorization_sha256,
        "fold_provenance": fold_provenance,
    }


def analyze(loaded: dict[str, Any], output: Path, protocol_sha256: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    setups = loaded["setups"]
    folds = loaded["folds"]
    unique_setups = np.unique(setups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(unique_setups), size=(BOOTSTRAP_RESAMPLES, len(unique_setups)), endpoint=False, dtype=np.int64)
    draw_hash = hashlib.sha256(np.ascontiguousarray(draws, dtype="<i8").tobytes()).hexdigest()
    if draw_hash != DRAW_SHA256:
        raise RuntimeError("Frozen bootstrap draw hash mismatch")
    draw_counts = np.stack([np.bincount(row, minlength=len(unique_setups)) for row in draws]).astype(np.float64)
    alpha = FAMILY_ALPHA / len(REGIMES)
    regime_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    regime_results: dict[str, Any] = {}
    for regime in REGIMES:
        full = loaded["medians"][regime][SYSTEMS[0]]
        stream3 = loaded["medians"][regime][SYSTEMS[1]]
        setup_full = np.asarray([full["latency_ms"][setups == setup].sum() for setup in unique_setups])
        setup_three = np.asarray([stream3["latency_ms"][setups == setup].sum() for setup in unique_setups])
        distribution = 1.0 - (draw_counts @ setup_three) / (draw_counts @ setup_full)
        estimate = paired_saving(full["latency_ms"], stream3["latency_ms"])
        lcb = basic_lcb(estimate, distribution, alpha)
        p_value = margin_p_value(estimate, distribution, MIN_SAVING)
        passed = bool(lcb > MIN_SAVING)
        row = {
            "cache_regime": regime,
            "n_samples": len(setups),
            "full4_mean_latency_ms": float(np.mean(full["latency_ms"])),
            "stream3_mean_latency_ms": float(np.mean(stream3["latency_ms"])),
            "measured_saving": estimate,
            "one_sided_lcb": lcb,
            "required_saving": MIN_SAVING,
            "multiplicity_alpha": alpha,
            "margin_null_p": p_value,
            "passed": passed,
        }
        regime_rows.append(row)
        regime_results[regime] = dict(row)
        for field in FIELDS:
            stage_rows.append({
                "cache_regime": regime,
                "stage": field,
                "full4_mean_ms": float(np.mean(full[field])),
                "stream3_mean_ms": float(np.mean(stream3[field])),
                "relative_saving": paired_saving(full[field], stream3[field]),
            })
        for fold in FOLDS:
            mask = folds == fold
            fold_rows.append({
                "cache_regime": regime,
                "fold": fold,
                "n_samples": int(mask.sum()),
                "full4_mean_latency_ms": float(np.mean(full["latency_ms"][mask])),
                "stream3_mean_latency_ms": float(np.mean(stream3["latency_ms"][mask])),
                "measured_saving": paired_saving(full["latency_ms"][mask], stream3["latency_ms"][mask]),
            })
    passed = all(value["passed"] for value in regime_results.values())
    summary = {
        "schema": "aura-har.ksp-stream3-gpu-microbenchmark-summary.v1",
        "status": "GPU_LATENCY_MICROBENCHMARK_COMPLETE",
        "created_utc": now_utc(),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "source_protocol_id": SOURCE_PROTOCOL_ID,
        "candidate": "drop_bone_motion",
        "systems": list(SYSTEMS),
        "decision": "LATENCY_GATE_PASS" if passed else "LATENCY_GATE_FAIL",
        "latency_gate_passed": passed,
        "quality_confirmatory": False,
        "quality_confirmation_requires_new_seed": True,
        "regimes": regime_results,
        "bootstrap": {
            "cluster": "setup",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "draw_sha256": draw_hash,
            "family_alpha": FAMILY_ALPHA,
            "per_regime_alpha": alpha,
        },
        "matrix": {"folds": 3, "cache_regimes": 2, "process_repetitions": 3, "samples_per_setup": 100, "traces": 18},
        "device": loaded["device"],
        "device_name": loaded["device_name"],
        "authorization_sha256": loaded["authorization_sha256"],
        "trace_hashes": loaded["trace_hashes"],
        "constraints": {"training": False, "test": False, "old_validation": False, "external_confirmation": False},
    }
    write_csv(output / "01_regime_latency_gate.csv", regime_rows)
    write_csv(output / "02_fold_regime_latency.csv", fold_rows)
    write_csv(output / "03_stage_timing.csv", stage_rows)
    write_json(output / "GPU_MICROBENCH_SUMMARY.json", summary)
    lines = [
        "# AURA-KSP: результат GPU latency microbenchmark",
        "",
        "Этот этап измеряет только стоимость существующих K64 checkpoints. Обучение и оценка качества не выполнялись.",
        "",
        f"Итоговый latency gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "| Cache regime | Full-4, ms | Stream-3, ms | Saving | LCB | Требование | Результат |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in regime_rows:
        lines.append(
            f"| {row['cache_regime']} | {row['full4_mean_latency_ms']:.4f} | "
            f"{row['stream3_mean_latency_ms']:.4f} | {100*row['measured_saving']:.2f}% | "
            f"{100*row['one_sided_lcb']:.2f}% | 20.00% | {'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "PASS разрешает заморозить отдельный confirmatory quality protocol. Он не является подтверждением качества сам по себе.",
        "Независимое подтверждение качества требует нового seed и девяти обучений: 3 folds × 3 retained streams.",
    ])
    (output / "REPORT_RU.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if path.name == "MANIFEST_SHA256.txt" or not path.is_file():
            continue
        manifest_rows.append(f"{sha256_file(path)}  {path.name}")
    (output / "MANIFEST_SHA256.txt").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol_file.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Protocol file mismatch")
    protocol_sha256 = sha256_file(protocol_path)
    loaded = load_traces(args.trace_root.resolve(), protocol_sha256)
    result = analyze(loaded, args.output.resolve(), protocol_sha256)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
