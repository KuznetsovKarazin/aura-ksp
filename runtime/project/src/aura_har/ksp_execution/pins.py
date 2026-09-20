#!/usr/bin/env python3
"""Read-only integrity check of the EXISTING AURA-KSP v0.3.3 CPU artifacts.

Python 3.9+, standard library only. Does not import/execute AURA code, install
packages, rebuild data/folds, train/evaluate models, create a lock, or authorize GPU.
Only a new, uniquely named TXT report is written, next to this script by default.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import itertools
import json
import logging
import re
import struct
import sys
import tarfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

VERSION = "1.0.1"
logger = logging.getLogger(__name__)
PROTOCOL_ID = "aura-ksp-ntu120-xset-trainonly-sparse-v1"
CODE_SHA = "92efb974d18c6a5eb8b30f4161c81d6dde4378a62e323a83b219f97cf5e5ad22"
AUDIT_SHA = "6f15f2454c73b1c64f74a83b0d188ec709fa4d0cdcb15464d1d642c75eab6b7a"
SOURCE_SHA = "fd037a9f71dbb00778bf71428892c8de407577402ee8330fdba0b02cea20edaa"
FRAMES_SHA = "3cec30ca30661b49e6ee7b86ce472e5e20890ec13ceb657c235be721c83ced8d"
TPC_LOCK_SHA = "74796b903bfc6f702541b3a841a74648baaf4d34c2de2a8a7e01798ad66ed305"
CODE_PREFIX = "AURA-HAR_v0.3.3/"
RUN_ROOT = "runs/aura_ksp_ntu120_xset_trainonly_sparse"
STORE_ROOT = "data/processed/ntu120_xset_sparse_v1"
FOLD_ROOT = RUN_ROOT + "/folds"
RESOLVED_ROOT = RUN_ROOT + "/resolved_configs"
SOURCE_PATH = "data/processed/ntu120_xset/manifests/train.jsonl"
PROTOCOL_PATH = "configs/ksp/ntu120_xset_trainonly_sparse.yaml"
TPC_LOCK_PATH = "protocols/AURA_TPC_NTU120_NEWCLASSES_XSET_LOCK.json"
SETUPS = (4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32)
HELDOUT = ((4, 12, 18, 26), (6, 14, 22, 28), (8, 16, 24, 32))
COUNTS = ((26105, 12565), (26144, 12526), (25091, 13579))
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
VARIANTS = ("k8_exact_uniform", "k16_exact_uniform", "k32_exact_uniform", "k64_resize_reference")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
NTU_ID = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\Z")
MAX_MEMBER = 64 * 1024 * 1024
MAX_UNCOMPRESSED = 256 * 1024 * 1024
PINS = {
    STORE_ROOT + "/store.json": (796, "a2f58252b9d4e45f4c295ba84e4a3900880205443257abb0fbe907c1c2fc7957"),
    STORE_ROOT + "/audit.json": (1484, "ea4b49b3d3b9584cafa998f15a78cccd48367b834234858902c7cad6177839b1"),
    STORE_ROOT + "/index.jsonl": (9808388, "e8c70ef75ce32dfd076d8cb2d26a0c15e8116f5535e691830706e4e41367ede5"),
    FOLD_ROOT + "/fold_plan.json": (3112, "1ad3090e1324d783253c298d92dcb77bd3f18c7bed115225fe2b528e86d04dc6"),
    RESOLVED_ROOT + "/stage1_plan.json": (168957, "74eed673c7d24d5d67ed45680ccd2c49ba7978b7dd192a3a08af932c75020c30"),
    RESOLVED_ROOT + "/train_commands.jsonl": (20018, "b20c62a328f3ff8803e9b1c9d8cddd06cdde3b3dcc7d88046dd4164813dbc4db"),
    RESOLVED_ROOT + "/evaluation_commands.jsonl": (37058, "5c94848eb2645d3ec8d3ec487cc90e2248bf805a4710094032cbb67805fd7bc0"),
    RESOLVED_ROOT + "/benchmark_commands.jsonl": (36662, "f4fbfebcd4334d7ae842d589bac935247f36c79df86b57b64354aac41a896cb0"),
}


class CheckError(ValueError):
    """A recorded integrity condition did not hold; no repair is attempted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_relative(value: str) -> str:
    name = str(value).replace("\\", "/")
    parts = name.split("/")
    require(bool(name) and "\x00" not in name, "Empty/NUL path")
    require(not PurePosixPath(name).is_absolute(), f"Absolute path is forbidden: {name}")
    require(all(p not in {"", ".", ".."} and ":" not in p for p in parts),
            f"Unsafe relative path: {name}")
    return name


def local_path(root: Path, relative: str) -> Path:
    target = (root / safe_relative(relative)).resolve()
    require(target.is_relative_to(root.resolve()),
            f"Path/junction escapes the explicitly selected project root: {relative}")
    return target


def sha_file(path: Path, *, progress: bool = False) -> str:
    before = path.stat()
    hasher = hashlib.sha256()
    processed = 0
    next_notice = 256 * 1024 * 1024
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
            processed += len(chunk)
            if progress and processed >= next_notice:
                print(f"  frames.npy: {processed / 1e6:.0f}/{before.st_size / 1e6:.0f} MB hashed", flush=True)
                next_notice += 256 * 1024 * 1024
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino) and processed == before.st_size,
            f"File changed while it was being hashed: {path}")
    return hasher.hexdigest()


def read_small(path: Path) -> bytes:
    require(path.stat().st_size <= MAX_MEMBER, f"Metadata file exceeds limit: {path}")
    with path.open("rb") as handle:
        raw = handle.read(MAX_MEMBER + 1)
    require(len(raw) <= MAX_MEMBER, f"Metadata file exceeds limit: {path}")
    return raw


def _unique_object(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str):
    raise CheckError(f"Non-finite JSON constant: {value}")


def json_value(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def json_rows(raw: bytes) -> list[dict]:
    rows = [json_value(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(all(isinstance(row, dict) for row in rows), "Every JSONL row must be an object")
    return rows


def decode_command_text(raw: bytes, *, source: str) -> tuple[str, str]:
    """Decode a saved command receipt without changing any source bytes.

    UTF-16/32 is accepted only with an explicit BOM. Data manifests continue to
    use the original UTF-8-only reader and unchanged canonical-jsonl-v1 rule.
    Check the longer UTF-32 marks before the UTF-16 prefix they can share.
    No guessing, replacement characters, or byte stripping is used.
    """
    variants = (
        (b"\xff\xfe\x00\x00", "utf-32-le", "utf-32-le-bom"),
        (b"\x00\x00\xfe\xff", "utf-32-be", "utf-32-be-bom"),
        (b"\xef\xbb\xbf", "utf-8", "utf-8-bom"),
        (b"\xff\xfe", "utf-16-le", "utf-16-le-bom"),
        (b"\xfe\xff", "utf-16-be", "utf-16-be-bom"),
    )
    codec, label, skip = "utf-8", "utf-8", 0
    for mark, candidate, name in variants:
        if raw.startswith(mark):
            codec, label, skip = candidate, name, len(mark)
            break
    try:
        text = raw[skip:].decode(codec, errors="strict")
    except UnicodeDecodeError as exc:
        raise CheckError(
            f"{source}: invalid {label} at byte {skip + exc.start}; "
            f"first bytes={raw[:8].hex()}. No source file was changed."
        ) from exc
    require("\x00" not in text, f"{source}: unexpected NUL after {label} decoding")
    return text, label


def json_command_rows(raw: bytes, *, source: str) -> tuple[list[list[str]], dict]:
    """Read JSON arrays of argv, NOT object-shaped dataset-manifest rows."""
    text, encoding = decode_command_text(raw, source=source)
    rows = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json_value(line)
        except (ValueError, TypeError) as exc:
            raise CheckError(f"{source}: invalid JSON at decoded line {number}: {exc}") from exc
        require(isinstance(row, list) and bool(row),
                f"{source}: decoded line {number} must be a nonempty argv array")
        require(all(isinstance(value, str) and bool(value) and
                    not any(c in value for c in ("\x00", "\r", "\n")) for value in row),
                f"{source}: decoded line {number} has an invalid argv token")
        rows.append(row)
    # This reproduces the OLD exporter's binary-line diagnostic, not JSON count.
    binary_lines = sum(1 for _ in io.BytesIO(raw))
    return rows, {
        "path": source, "encoding": encoding, "raw_prefix_hex": raw[:8].hex(),
        "raw_bytes": len(raw), "raw_sha256": digest(raw),
        "binary_readline_records": binary_lines,
        "decoded_json_records": len(rows), "row_schema": "argv_array_of_strings",
        "source_bytes_modified": False,
    }


def check_command_exports(plan: dict, cpu: dict[str, bytes]) -> dict:
    """Check the actual decoded commands against the saved plan, in order."""
    expected_counts = {"train_commands": 48, "evaluation_commands": 48,
                       "benchmark_commands": 30}
    result = {}
    for name, expected_count in expected_counts.items():
        path = RESOLVED_ROOT + "/" + name + ".jsonl"
        require(path in cpu, f"Command receipt missing: {path}")
        raw = cpu[path]
        size, pinned_sha = PINS[path]
        require(len(raw) == size and digest(raw) == pinned_sha,
                f"Pinned command receipt bytes changed: {path}")
        commands, details = json_command_rows(raw, source=path)
        require(len(commands) == expected_count,
                f"{path}: expected {expected_count} decoded commands, got {len(commands)} "
                f"({details['encoding']}; binary fragments are not JSON records)")
        require(commands == plan.get(name),
                f"{path}: decoded argv arrays do not match stage1_plan.json in exact order")
        result[name] = {**details, "raw_pin_verified": True,
                        "matches_stage1_plan_in_order": True}
    return result


def canonicalize(value, key=None):
    # Exact canonical-jsonl-v1 rule from the recovered src/aura_har/utils/io.py.
    if isinstance(value, dict):
        return {name: canonicalize(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, str) and key is not None and key.endswith("_path"):
        return value.replace("\\", "/")
    return value


def canonical_hash(rows: list[dict]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        line = json.dumps(canonicalize(row), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
        hasher.update(line.encode("utf-8") + b"\n")
    return hasher.hexdigest()


def normalize_path(value) -> str:
    return str(value).replace("\\", "/")


def checked_source(rows: list[dict], setups=SETUPS) -> dict[str, dict]:
    result = {}
    for row in rows:
        sid = row.get("sample_id")
        require(isinstance(sid, str), "Missing/string-invalid source sample_id")
        match = NTU_ID.fullmatch(sid)
        require(match is not None, f"Invalid NTU sample_id: {sid}")
        require(sid not in result, f"Duplicate source sample: {sid}")
        require(type(row.get("setup")) is int and row["setup"] in setups,
                f"Forbidden or invalid setup: {sid}")
        require(int(match[1]) == row["setup"], f"Setup/sample_id mismatch: {sid}")
        require(type(row.get("label")) is int and 0 <= row["label"] < 120 and
                int(match[5]) - 1 == row["label"], f"Label/sample_id mismatch: {sid}")
        require(type(row.get("sequence_length")) is int and row["sequence_length"] > 0,
                f"Invalid sequence_length: {sid}")
        require(row.get("split", "train") == "train", f"Non-train source row: {sid}")
        safe_relative(row["data_path"])
        for field, group in (("camera", 2), ("subject", 3), ("repetition", 4)):
            if field in row:
                require(int(row[field]) == int(match[group]), f"{field}/sample_id mismatch: {sid}")
        result[sid] = row
    require({row["setup"] for row in rows} == set(setups), "Source setup inventory mismatch")
    return result


def check_index(source: list[dict], index: list[dict], total_frames: int) -> dict:
    require(len(index) == len(source), "Index/source sample count mismatch")
    offset = 0
    for src, item in zip(source, index):
        sid = src["sample_id"]
        require(item.get("sample_id") == sid, f"Index/source order mismatch: {sid}")
        for field in ("label", "setup", "sequence_length"):
            require(type(item.get(field)) is int and item[field] == src[field],
                    f"Index/source {field} mismatch: {sid}")
        require(type(item.get("frame_offset")) is int and item["frame_offset"] == offset,
                f"Non-contiguous index offset: {sid}")
        require(normalize_path(item.get("source_data_path")) == normalize_path(src["data_path"]),
                f"Index/source data path mismatch: {sid}")
        require(isinstance(item.get("content_sha256"), str) and
                HEX64.fullmatch(item["content_sha256"]) is not None,
                f"Invalid semantic hash in index: {sid}")
        offset += src["sequence_length"]
    require(offset == total_frames, "Index total frame count mismatch")
    return {"samples": len(index), "total_frames": offset, "ordered_source_equivalence": True,
            "semantic_frame_hashes_recomputed": False}


def check_fold_rows(source: list[dict], partitions: list[dict], heldout_sets=HELDOUT) -> list[dict]:
    source_by_id = {row["sample_id"]: row for row in source}
    require(len(source_by_id) == len(source), "Duplicate source ID")
    require(len(partitions) == len(heldout_sets), "Wrong number of folds")
    heldout_counter = Counter()
    results = []
    for number, (part, heldout_setups) in enumerate(zip(partitions, heldout_sets), 1):
        split_ids = {}
        for role in ("train", "heldout"):
            rows = part[role]
            ids = [row["sample_id"] for row in rows]
            require(len(ids) == len(set(ids)), f"Duplicate ID in fold {number} {role}")
            expected = {row["sample_id"] for row in source
                        if (row["setup"] in heldout_setups) == (role == "heldout")}
            require(set(ids) == expected, f"Exact source partition mismatch: fold {number} {role}")
            for row in rows:
                src = source_by_id[row["sample_id"]]
                for field in ("label", "setup", "sequence_length", "data_path"):
                    a = normalize_path(row.get(field)) if field == "data_path" else row.get(field)
                    b = normalize_path(src[field]) if field == "data_path" else src[field]
                    require(a == b, f"Fold/source {field} mismatch: fold {number} {row['sample_id']}")
            split_ids[role] = set(ids)
        require(not split_ids["train"] & split_ids["heldout"], f"Fold {number} train/heldout overlap")
        require(split_ids["train"] | split_ids["heldout"] == set(source_by_id),
                f"Fold {number} incomplete coverage")
        heldout_counter.update(split_ids["heldout"])
        results.append({"fold": number, "train_samples": len(part["train"]),
                        "heldout_samples": len(part["heldout"]),
                        "heldout_setups": list(heldout_setups), "sample_overlap": 0,
                        "exact_source_partition": True})
    require(heldout_counter == Counter({sid: 1 for sid in source_by_id}),
            "Heldout coverage is not exactly once")
    return results


def check_npy_header(path: Path, frames: int) -> dict:
    with path.open("rb") as handle:
        require(handle.read(6) == b"\x93NUMPY", "Invalid NPY magic")
        version = handle.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        width = 2 if version[0] == 1 else 4
        size_raw = handle.read(width)
        require(len(size_raw) == width, "Truncated NPY header length")
        size = struct.unpack("<H" if width == 2 else "<I", size_raw)[0]
        require(0 < size <= 65536, "Unreasonable NPY header length")
        raw = handle.read(size)
        require(len(raw) == size, "Truncated NPY header")
        header = ast.literal_eval(raw.decode("utf-8" if version[0] == 3 else "latin-1"))
        require(isinstance(header, dict), "NPY header is not a dict")
        require(header.get("descr") == "<f4" and header.get("fortran_order") is False,
                "Expected C-contiguous little-endian float32 NPY")
        require(header.get("shape") == (frames, 3, 25, 2), "NPY shape mismatch")
        offset = handle.tell()
    logical = frames * 600
    require(path.stat().st_size == offset + logical, "NPY file size/payload mismatch")
    return {"shape": [frames, 3, 25, 2], "dtype": "<f4", "header_bytes": offset,
            "logical_bytes": logical, "file_bytes": offset + logical}


def choose_archive(explicit: str | None, directories: list[Path], pattern: str, expected: str) -> Path:
    if explicit:
        paths = [Path(explicit).expanduser().resolve()]
    else:
        paths = sorted({p.resolve() for root in directories if root.is_dir()
                        for p in root.glob(pattern) if p.is_file()}, key=str)
    require(bool(paths), f"Archive not found: {pattern}. Specify --audit / --code-archive explicitly.")
    mismatches = []
    for path in paths:
        value = sha_file(path)
        if value == expected:
            return path
        mismatches.append({"path": str(path), "sha256": value})
    raise CheckError(f"No archive matches the historical SHA-256: {json.dumps(mismatches)}")


def read_code_archive(path: Path) -> dict[str, bytes]:
    result = {}
    total = 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = safe_relative(entry.filename)
            require(name.startswith(CODE_PREFIX), f"Unexpected code ZIP root: {name}")
            relative = name[len(CODE_PREFIX):]
            safe_relative(relative)
            require(relative not in result, f"Duplicate code ZIP entry: {relative}")
            require(((entry.external_attr >> 16) & 0o170000) != 0o120000, f"ZIP symlink: {relative}")
            require(0 <= entry.file_size <= MAX_MEMBER, f"Oversize ZIP entry: {relative}")
            total += entry.file_size
            require(total <= MAX_UNCOMPRESSED, "Code ZIP exceeds decompression limit")
            # Reading each member to EOF invokes the ZIP CRC check.
            result[relative] = archive.read(entry)
    require(len(result) == 273, f"Code ZIP file count changed: {len(result)}")
    return result


def read_audit_archive(path: Path) -> dict[str, bytes]:
    # Complete gzip read checks the trailer; the earlier text exporter did not claim this.
    total = 0
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            total += len(block)
            require(total <= MAX_UNCOMPRESSED, "CPU tar/gzip exceeds decompression limit")
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for entry in archive:
            if entry.isdir():
                continue
            name = safe_relative(entry.name)
            require(entry.isfile(), f"Non-regular TAR entry: {name}")
            require(name not in result, f"Duplicate TAR entry: {name}")
            require(0 <= entry.size <= MAX_MEMBER, f"Oversize TAR entry: {name}")
            stream = archive.extractfile(entry)
            require(stream is not None, f"Cannot read TAR entry: {name}")
            with stream:
                raw = stream.read(MAX_MEMBER + 1)
            require(len(raw) == entry.size, f"TAR member size mismatch: {name}")
            result[name] = raw
    require(len(result) == 114, f"CPU TAR file count changed: {len(result)}")
    for name, (size, expected) in PINS.items():
        require(name in result and len(result[name]) == size and digest(result[name]) == expected,
                f"Pinned CPU metadata mismatch: {name}")
    return result


def compare_local(root: Path, entries: dict[str, bytes], *, group: str) -> dict:
    missing, changed = [], []
    for relative, expected in entries.items():
        path = local_path(root, relative)
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size != len(expected) or sha_file(path) != digest(expected):
            changed.append(relative)
    require(not missing and not changed,
            f"{group} differs from its saved archive. Missing={missing}; changed={changed}. "
            "No files were changed or repaired.")
    return {"files": len(entries), "all_match_saved_archive_bytes": True}


def check_tpc_lock(code: dict[str, bytes]) -> dict:
    raw = code[TPC_LOCK_PATH]
    require(digest(raw) == TPC_LOCK_SHA, "Historical TPC lock hash mismatch")
    lock = json_value(raw)
    candidates = [(key, value) for key, value in lock.items() if isinstance(value, dict)
                  and len(value) == 29 and all(isinstance(v, str) and HEX64.fullmatch(v)
                                              for v in value.values())]
    require(len(candidates) == 1, "Cannot identify the unique 29-file TPC lock mapping")
    key, entries = candidates[0]
    for relative, expected in entries.items():
        name = safe_relative(relative)
        require(name in code and digest(code[name]) == expected, f"TPC locked file changed: {name}")
    return {"lock_sha256": TPC_LOCK_SHA, "mapping_key": key, "locked_files": 29, "mismatches": 0}


def check_stage1(plan: dict, folds: list[dict], cpu: dict[str, bytes], code: dict[str, bytes]) -> dict:
    require(plan.get("schema") == "aura-har.ksp-command-plan.v1", "Plan schema mismatch")
    require(plan.get("protocol_id") == PROTOCOL_ID, "Plan protocol mismatch")
    for key in ("executable", "gpu_authorized", "test_read", "old_validation_used", "external_confirmation"):
        require(plan.get(key) is False, f"Unexpected plan state: {key}")
    require(plan.get("base_seeds") == [271828] and plan.get("mode") == "stage1_train_only", "Plan scope changed")
    require(plan.get("protocol_status") == "draft_before_data_audit_and_gpu", "Plan status changed")
    require(plan.get("primary_candidate") == "k32_exact_uniform" and
            plan.get("primary_candidate_preselected") is True and
            plan.get("secondary_candidates_cannot_rescue_primary") is True, "Primary K32 rule changed")
    require(plan.get("protocol_config_sha256") == digest(code[PROTOCOL_PATH]), "Plan/protocol hash mismatch")
    bindings = {
        "fold_plan_sha256": FOLD_ROOT + "/fold_plan.json",
        "sparse_store_audit_sha256": STORE_ROOT + "/audit.json",
        "sparse_store_metadata_sha256": STORE_ROOT + "/store.json",
        "sparse_store_index_sha256": STORE_ROOT + "/index.jsonl",
    }
    for key, relative in bindings.items():
        require(plan.get(key) == digest(cpu[relative]), f"Plan binding changed: {key}")
    require(plan.get("sparse_store_frames_sha256") == FRAMES_SHA, "Plan/frame binding changed")
    wanted = set(itertools.product((1, 2, 3), VARIANTS, STREAMS))
    seen_configs = set()
    for kind, commands_key, role, prefix in (
        ("trainings", "train_commands", "train", "train"),
        ("evaluations", "evaluation_commands", "heldout", "evaluate"),
    ):
        records = plan["artifacts"][kind]
        require(len(records) == 48 and len(plan[commands_key]) == 48, f"Wrong {kind} count")
        seen = set()
        for record in records:
            key = (record["fold"], record["variant"], record["stream"])
            require(key in wanted and key not in seen, f"Missing/duplicate/unexpected {kind} job: {key}")
            seen.add(key)
            number, _variant, stream = key
            fold = folds[number - 1]
            require(record.get("outer_fold") == number and record.get("role") == role, "Wrong job fold/role")
            require(record.get("base_seed") == 271828 and
                    record.get("job_seed") == 271828 + 1000 * (number - 1) + 10 * STREAMS.index(stream),
                    "Unpaired/incorrect KSP job seed")
            for record_key, fold_key in (("manifest_sha256", f"{role}_manifest_sha256"),
                                         ("manifest_samples", f"{role}_samples"),
                                         ("manifest_setups", f"{role}_setups")):
                require(record.get(record_key) == fold[fold_key], f"Wrong job binding: {record_key}")
            require(normalize_path(record["manifest"]) == normalize_path(fold[f"{role}_manifest"]),
                    "Job manifest path changed")
            relative = safe_relative(record["config"])
            require(relative in cpu and relative not in seen_configs, "Missing/duplicate resolved config")
            require(relative.endswith(f"/{prefix}_{stream}.yaml"), "Wrong job config filename")
            seen_configs.add(relative)
            require(normalize_path(record["checkpoint"]).endswith(f"/checkpoints/{stream}/final.pt"),
                    "Job does not use a final.pt checkpoint")
        require(seen == wanted, f"Incomplete {kind} Cartesian matrix")
        command_configs = []
        for argv in plan[commands_key]:
            require(isinstance(argv, list) and argv[:2] == ["python", f"scripts/{prefix}_ksp.py"],
                    "Unexpected planned executable")
            require(argv.count("--config") == 1 and argv.count("--device") == 1 and
                    argv[argv.index("--device") + 1] == "cuda", "Unexpected job command structure")
            command_configs.append(safe_relative(argv[argv.index("--config") + 1]))
        require(Counter(command_configs) == Counter(safe_relative(r["config"]) for r in records),
                "Command/artifact config coverage mismatch")
    benchmarks = plan["artifacts"]["benchmarks"]
    wanted_cost = set(itertools.product((1, 2, 3), ("warm_reuse", "fresh_mapping"), (1, 2, 3, 4, 5)))
    require(len(benchmarks) == 30 and len(plan["benchmark_commands"]) == 30, "Wrong cost job count")
    cost_keys = [(r["fold"], r["cache_regime"], r["replicate"]) for r in benchmarks]
    require(len(set(cost_keys)) == 30 and set(cost_keys) == wanted_cost, "Cost matrix incomplete/duplicated")
    require(all(r["base_seed"] == 271828 and r["variants"] == list(VARIANTS) for r in benchmarks),
            "Cost job seed/variant mismatch")
    absolute_protocol_paths = sorted({str(argv[argv.index("--protocol-config") + 1])
                                     for argv in plan["benchmark_commands"]})
    windows_absolute = [p for p in absolute_protocol_paths if re.match(r"^[A-Za-z]:[\\/]", p)]
    command_files = check_command_exports(plan, cpu)
    counts = {name: entry["decoded_json_records"] for name, entry in command_files.items()}
    return {"trainings": 48, "evaluations": 48, "benchmarks": 30,
            "resolved_configs_byte_verified": len(seen_configs),
            "yaml_semantic_validation_executed": False,
            "command_jsonl_records": counts, "command_files": command_files,
            "windows_absolute_protocol_paths": windows_absolute,
            "linux_rebinding_required": bool(windows_absolute), "execution_authorized": False}


def perform(args, report: dict) -> None:
    root = Path(args.project_root).expanduser().resolve()
    require(root.is_dir(), f"Project root not found: {root}")
    report["project_root"] = str(root)
    locations = list(dict.fromkeys([root, root.parent, root.parent.parent, Path.cwd(),
                                   Path(__file__).resolve().parent, Path.home() / "Downloads", Path("E:/AURA")]))
    def passed(name: str, details: dict) -> None:
        report["checks"].append({"check": name, "status": "PASS", **details})
        print(f"PASS: {name}", flush=True)

    report["current_check"] = "saved_archives"
    print("Locating the historical code and CPU-audit archives...", flush=True)
    code_path = choose_archive(args.code_archive, locations, "AURA-HAR_v0.3.3_portable*.zip", CODE_SHA)
    audit_path = choose_archive(args.audit, locations, "AURA-KSP_CPU_AUDIT_v1*.gz", AUDIT_SHA)
    code = read_code_archive(code_path)
    cpu = read_audit_archive(audit_path)
    passed("saved_archives", {"code_path": str(code_path), "code_sha256": CODE_SHA,
                              "audit_path": str(audit_path), "audit_sha256": AUDIT_SHA,
                              "code_files": len(code), "audit_files": len(cpu),
                              "all_zip_member_crc_checked": True, "gzip_trailer_checked": True})

    report["current_check"] = "existing_local_code_and_cpu_metadata"
    passed("existing_local_code", compare_local(root, code, group="Local v0.3.3 code"))
    # The root checksum note is a packaging convenience, not a runtime input;
    # it may have been created only in the temporary archive-staging directory.
    runtime_cpu = {k: v for k, v in cpu.items() if k != "AURA_KSP_FRAMES_SHA256.txt"}
    passed("existing_local_cpu_metadata", compare_local(root, runtime_cpu, group="Local saved CPU metadata"))
    passed("legacy_tpc_boundary", check_tpc_lock(code))

    report["current_check"] = "saved_deep_audit_bindings"
    store = json_value(cpu[STORE_ROOT + "/store.json"])
    audit = json_value(cpu[STORE_ROOT + "/audit.json"])
    plan = json_value(cpu[FOLD_ROOT + "/fold_plan.json"])
    require(audit["status"] == "passed" and audit["deep"] is True, "Saved source audit did not pass")
    require(len(audit["checks"]) == 10 and all(value is True for value in audit["checks"].values()),
            "Saved deep audit check set changed")
    for field in ("test_read", "old_validation_used", "gpu_used"):
        require(audit.get(field) is False, f"Saved audit has unexpected {field}")
    require(store["samples"] == audit["samples"] == plan["source_samples"] == 38670, "Sample count mismatch")
    require(store["setups"] == audit["setups"] == plan["source_setups"] == list(SETUPS), "Setup boundary changed")
    require(store["source_manifest_sha256"] == audit["source_manifest_sha256"] ==
            plan["source_manifest_sha256"] == SOURCE_SHA, "Source manifest binding changed")
    require(store["frames_sha256"] == audit["store_frames_sha256"] == FRAMES_SHA, "Frame binding changed")
    require(store["total_frames"] == audit["total_frames"] == 2802727, "Frame count changed")
    require(store["logical_data_bytes"] == 600 * store["total_frames"] == audit["logical_data_bytes"],
            "Logical payload size mismatch")
    passed("saved_deep_audit_bindings", {"saved_status": "passed", "saved_checks": 10,
                                         "source_npz_equivalence_reexecuted": False})

    report["current_check"] = "canonical_source_manifest_and_sparse_index"
    source_path = local_path(root, SOURCE_PATH)
    require(source_path.name == "train.jsonl", "Source manifest resolves to a different role")
    source = json_rows(read_small(source_path))
    require(canonical_hash(source) == SOURCE_SHA and len(source) == 38670, "Canonical source hash/count mismatch")
    checked_source(source)
    index = json_rows(cpu[STORE_ROOT + "/index.jsonl"])
    index_result = check_index(source, index, store["total_frames"])
    passed("canonical_source_manifest_and_sparse_index", {"canonical_sha256": SOURCE_SHA, **index_result})

    report["current_check"] = "exact_fold_coverage"
    partitions = []
    require([f["fold"] for f in plan["folds"]] == [1, 2, 3], "Fold order changed")
    for i, fold in enumerate(plan["folds"]):
        require(fold["heldout_setups"] == list(HELDOUT[i]), "Heldout setup design changed")
        require((fold["train_samples"], fold["heldout_samples"]) == COUNTS[i], "Fold sample counts changed")
        part = {}
        for role in ("train", "heldout"):
            name = safe_relative(fold[f"{role}_manifest"])
            part[role] = json_rows(cpu[name])
            require(len(part[role]) == fold[f"{role}_samples"] and
                    canonical_hash(part[role]) == fold[f"{role}_manifest_sha256"],
                    f"Canonical fold hash/count mismatch: {name}")
        partitions.append(part)
    folds = check_fold_rows(source, partitions)
    passed("exact_fold_coverage", {"folds": folds, "heldout_coverage": "exactly_once",
                                    "six_canonical_fold_hashes_recomputed": True})

    report["current_check"] = "stage1_plan_integrity"
    stage1 = json_value(cpu[RESOLVED_ROOT + "/stage1_plan.json"])
    passed("stage1_plan_integrity", check_stage1(stage1, plan["folds"], cpu, code))

    report["current_check"] = "existing_train_frame_array"
    require(store["frames_path"] == "frames.npy", "Unexpected frame payload name")
    frames_path = local_path(root, STORE_ROOT + "/frames.npy")
    print("Checking the existing train-only frames.npy header and SHA-256 (no rebuild)...", flush=True)
    header = check_npy_header(frames_path, store["total_frames"])
    actual_frames_sha = sha_file(frames_path, progress=True)
    require(actual_frames_sha == FRAMES_SHA, "Existing frames.npy SHA-256 mismatch")
    passed("existing_train_frame_array", {**header, "sha256": actual_frames_sha,
                                          "source_npz_opened": False})
    report["current_check"] = "archive_stability"
    require(sha_file(code_path) == CODE_SHA and sha_file(audit_path) == AUDIT_SHA,
            "An input archive changed during the integrity check")
    passed("archive_stability", {"archive_hashes_unchanged_at_completion": True})
    report.pop("current_check", None)
    report["status"] = "PRELOCK_INTEGRITY_PASS_NO_GPU_AUTHORIZATION"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=r"E:\AURA\ksp-v033\AURA-HAR_v0.3.3")
    parser.add_argument("--audit", help="Exact CPU-audit .tar.gz path; normally autodetected")
    parser.add_argument("--code-archive", help="Exact original v0.3.3 ZIP path; normally autodetected")
    parser.add_argument("--output-dir", help="Report directory; default is the script directory")
    args = parser.parse_args(argv)
    print(f"AURA-KSP read-only prelock checker {VERSION}", flush=True)
    started = time.perf_counter()
    report = {
        "schema": "aura-har.readonly-prelock-integrity.v1", "checker_version": VERSION,
        "checker_sha256": sha_file(Path(__file__)),
        "protocol_id": PROTOCOL_ID, "time_utc": datetime.now(timezone.utc).isoformat(),
        "status": "IN_PROGRESS", "checks": [], "project_modified": False,
        "project_code_executed": False, "preprocessing_reexecuted": False,
        "scientific_training_or_evaluation_run": False, "test_read": False,
        "source_npz_equivalence_reexecuted": False, "gpu_authorized": False,
        "protocol_lock_created": False,
        "limitations": [
            "Integrity checking is not new scientific evidence or GPU authorization.",
            "The saved deep source-NPZ equivalence audit is validated, not repeated.",
            "K8/K16/K32 sparse read bounds and K64 dense equivalence are not re-tested here.",
            "Resolved YAML files are byte-verified; AURA validators/tests are not executed.",
            "A portable execution binding, real protocol lock and separate user authorization remain necessary."
        ],
    }
    result = 0
    try:
        perform(args, report)
    except Exception as exc:
        logger.exception("Read-only prelock integrity check failed")
        result = 1
        report["status"] = "PRELOCK_INTEGRITY_BLOCKED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).resolve().parent
    try:
        root = Path(args.project_root).expanduser().resolve()
        require(not out_dir.is_relative_to(root), "Write the report OUTSIDE the audited project (--output-dir).")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        out = out_dir / f"AURA_KSP_PRELOCK_CHECK_{stamp}.txt"
        with out.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nSTATUS: {report['status']}\nREPORT: {out}", flush=True)
        print("Attach this one TXT report. No archives need to be uploaded again.", flush=True)
    except (OSError, ValueError) as exc:
        print(f"Could not save report: {exc}\n{json.dumps(report, ensure_ascii=False, indent=2)}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
