"""Content-preserving conversion of the accepted KSP draft into a portable build.

This module never authorizes execution and never opens a dataset/checkpoint.
Scientific configurations stay in STATE D. Serialization hashes are kept outside
objects whose bytes they identify, so the binding graph is acyclic.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from . import pins as _pins
from .pins import (
    AUDIT_SHA,
    CODE_SHA,
    FOLD_ROOT,
    FRAMES_SHA,
    PINS,
    PROTOCOL_ID,
    PROTOCOL_PATH,
    RESOLVED_ROOT,
    RUN_ROOT,
    SOURCE_SHA,
    STORE_ROOT,
    STREAMS,
    VARIANTS,
    check_command_exports,
    check_stage1,
    digest,
    json_value,
)

# Public re-exports keep the original objects without redundant import aliases.
SETUPS = _pins.SETUPS
TPC_LOCK_SHA = _pins.TPC_LOCK_SHA

VERSION = "1.0.0"
WINDOWS_PROTOCOL = (
    r"E:\AURA\ksp-v033\AURA-HAR_v0.3.3\configs\ksp\ntu120_xset_trainonly_sparse.yaml"
)
PLAN_PATH = RESOLVED_ROOT + "/stage1_plan.json"
FOLD_PLAN_PATH = FOLD_ROOT + "/fold_plan.json"
JOB_PLAN_PATH = "execution_preparation/BOUND_PLAN_CANDIDATE.json"
CANDIDATE_LOCK_PATH = "execution_preparation/PROTOCOL_LOCK_CANDIDATE.json"
PATH_KEYS = frozenset({
    "manifest", "source_manifest", "train_manifest", "heldout_manifest",
    "output_dir", "sparse_store_root", "sparse_store_audit", "config",
    "checkpoint", "predictions_path", "raw_timing_path", "fold_plan",
    "protocol_config", "protocol_config_path", "resolved_root",
})
PATH_FLAGS = frozenset({"--config", "--checkpoint", "--protocol-config", "--resolved-root", "--output"})
COMMAND_KINDS = (("trainings", "train_commands", "train"),
                 ("evaluations", "evaluation_commands", "evaluate"),
                 ("benchmarks", "benchmark_commands", "benchmark"))


class PreparationError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise PreparationError(message)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def object_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def relative_path(value: str) -> str:
    """Portable, non-ambiguous path: reject traversal, drive/UNC, NTFS aliases."""
    require(isinstance(value, str) and bool(value), "Path must be a nonempty string")
    name = value.replace("\\", "/")
    require(not name.startswith("/"), f"Absolute/UNC path rejected: {value!r}")
    parts = name.split("/")
    for part in parts:
        require(part not in {"", ".", ".."}, f"Noncanonical path: {value!r}")
        require(not any(ord(c) < 32 or c in ':*?<>|"' for c in part), f"Unsafe path: {value!r}")
        require(part == part.rstrip(" ."), f"Windows path alias: {value!r}")
        stem = part.split(".", 1)[0].upper()
        require(stem not in {"CON", "PRN", "AUX", "NUL"} and
                not re.fullmatch(r"(?:COM|LPT)[1-9]", stem), f"Reserved path: {value!r}")
    return name


def portable_path(value: str) -> str:
    if value == WINDOWS_PROTOCOL:
        return PROTOCOL_PATH
    return relative_path(value)


def path_under(root: Path, relative: str) -> Path:
    target = root / relative_path(relative)
    resolved = target.resolve()
    require(resolved.is_relative_to(root.resolve()), f"Path escapes root: {relative}")
    return target


class UniqueLoader(yaml.SafeLoader):
    """Do not silently overwrite duplicate YAML fields or accept merge aliases."""


def _unique_mapping(loader: UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        require(isinstance(key, str), "YAML mapping key must be a string")
        require(key not in result, f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _plain(value: Any, active: set[int] | None = None) -> None:
    active = set() if active is None else active
    if isinstance(value, (dict, list)):
        require(id(value) not in active, "Recursive YAML aliases are not supported")
        active.add(id(value))
        for item in (value.values() if isinstance(value, dict) else value):
            _plain(item, active)
        active.remove(id(value))
    else:
        require(value is None or type(value) in (str, int, float, bool), "Unsupported YAML value type")
        require(not isinstance(value, float) or math.isfinite(value), "Non-finite YAML value")


def load_yaml_bytes(raw: bytes) -> dict:
    value = yaml.load(raw.decode("utf-8"), Loader=UniqueLoader)
    require(isinstance(value, dict), "YAML root must be a mapping")
    _plain(value)
    return value


def convert_paths(value: Any, location: tuple = ()) -> tuple[Any, list[dict]]:
    """Convert only declared path fields. Unknown path-bearing fields fail closed."""
    changes = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PATH_KEYS:
                require(isinstance(item, str), f"Expected path string at {location + (key,)}")
                result[key] = portable_path(item)
                if result[key] != item:
                    changes.append({"field": list(location + (key,)), "old": item, "new": result[key]})
            else:
                result[key], extra = convert_paths(item, location + (key,))
                changes.extend(extra)
        return result, changes
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            converted, extra = convert_paths(item, location + (index,))
            result.append(converted)
            changes.extend(extra)
        return result, changes
    if isinstance(value, str):
        require("\\" not in value and not re.match(r"^[A-Za-z]:[/\\]", value),
                f"Unclassified path at {location}; do not guess: {value!r}")
    return value, changes


def convert_yaml(raw: bytes) -> tuple[bytes, list[dict]]:
    original = load_yaml_bytes(raw)
    converted, changes = convert_paths(original)
    serialized = yaml.safe_dump(converted, sort_keys=False, allow_unicode=True,
                                default_flow_style=False, width=120).encode("utf-8")
    require(load_yaml_bytes(serialized) == converted, "YAML round-trip changed semantic values")
    restored = copy.deepcopy(converted)
    for change in changes:
        ref = restored
        for key in change["field"][:-1]:
            ref = ref[key]
        ref[change["field"][-1]] = change["old"]
    require(restored == original, "A non-path YAML value changed")
    return serialized, changes


def convert_argv(argv: list[str]) -> tuple[list[str], list[dict]]:
    require(isinstance(argv, list) and len(argv) >= 2 and argv[0] == "python", "Unknown command form")
    require(all(isinstance(t, str) and t and not any(ord(c) < 32 for c in t) for t in argv),
            "Invalid command token")
    result = list(argv)
    changes = []
    for flag in PATH_FLAGS:
        count = argv.count(flag)
        require(count <= 1, f"Duplicate flag: {flag}")
        if count:
            index = argv.index(flag) + 1
            require(index < len(argv) and not argv[index].startswith("--"), f"Missing value: {flag}")
            result[index] = portable_path(argv[index])
            if result[index] != argv[index]:
                changes.append({"argument_index": index, "flag": flag,
                                "old": argv[index], "new": result[index]})
    result[1] = relative_path(result[1])
    require(result[1] in {"scripts/train_ksp.py", "scripts/evaluate_ksp.py", "scripts/benchmark_ksp_system.py"},
            "Unexpected scientific entry point")
    require(all("\\" not in token for token in result), "Unclassified Windows command path")
    return result, sorted(changes, key=lambda c: c["argument_index"])


def _expected_run_paths(fold: int, variant: str, stream: str, role: str) -> dict[str, str]:
    tail = f"stage1/seed_271828/outer_fold_{fold}/{variant}"
    return {
        "config": f"{RESOLVED_ROOT}/{tail}/{'train' if role == 'train' else 'evaluate'}_{stream}.yaml",
        "checkpoint": f"{RUN_ROOT}/{tail}/checkpoints/{stream}/final.pt",
        "manifest": f"{FOLD_ROOT}/outer_fold_{fold}/{role}.jsonl",
        "output_dir": f"{RUN_ROOT}/{tail}/{'checkpoints' if role == 'train' else 'predictions'}/{stream}",
        "predictions_path": f"{RUN_ROOT}/{tail}/predictions/{stream}/predictions.npz",
    }


def validate_job_alignment(plan: dict, config_bytes: dict[str, bytes], folds: dict) -> dict:
    """Independent exact correspondence: YAML <-> artifact <-> argv, all 126 jobs."""
    require(plan.get("protocol_id") == PROTOCOL_ID, "Protocol ID changed")
    require(plan.get("base_seeds") == [271828], "Stage-1 seed changed")
    require(plan.get("mode") == "stage1_train_only", "Stage-1 mode changed")
    require(plan.get("primary_candidate") == "k32_exact_uniform" and
            plan.get("primary_candidate_preselected") is True and
            plan.get("secondary_candidates_cannot_rescue_primary") is True, "Primary-K32 rule changed")
    for field in ("executable", "gpu_authorized", "old_validation_used", "test_read", "external_confirmation"):
        require(plan.get(field) is False, f"Unexpected execution flag: {field}")
    require(len(folds["folds"]) == 3, "Three frozen folds required")
    configs_seen = set()
    expected_cartesian = set(itertools.product((1, 2, 3), VARIANTS, STREAMS))
    for kind, section, action in COMMAND_KINDS[:2]:
        records, commands = plan["artifacts"][kind], plan[section]
        require(len(records) == len(commands) == 48, "Incomplete train/evaluation matrix")
        seen = set()
        role = "train" if action == "train" else "heldout"
        for record, argv in zip(records, commands, strict=True):
            key = record["fold"], record["variant"], record["stream"]
            require(key in expected_cartesian and key not in seen, f"Unexpected/duplicate job: {key}")
            seen.add(key)
            fold, variant, stream = key
            expected = _expected_run_paths(fold, variant, stream, role)
            seed = 271828 + 1000 * (fold - 1) + 10 * STREAMS.index(stream)
            require(record.get("base_seed") == 271828 and record.get("job_seed") == seed,
                    "Unpaired/incorrect job seed")
            require(record.get("outer_fold") == fold and record.get("role") == role, "Job role/fold mismatch")
            for field in ("config", "checkpoint", "manifest"):
                require(record.get(field) == expected[field], f"Artifact {field} mismatch")
            if role == "heldout":
                require(record.get("predictions_path") == expected["predictions_path"], "Prediction output mismatch")
            cfg = load_yaml_bytes(config_bytes[record["config"]])
            configs_seen.add(record["config"])
            require(cfg.get("schema") == "aura-har.ksp-run-config.v1", "Unexpected run config schema")
            require(cfg["experiment"]["output_dir"] == expected["output_dir"], "YAML output path mismatch")
            require(cfg["experiment"]["seed"] == seed, "YAML experiment seed mismatch")
            data = cfg["data"]
            for field, expected_value in (("manifest", expected["manifest"]), ("role", role),
                    ("outer_fold", fold), ("modality", stream), ("variant_id", variant),
                    ("source_manifest_sha256", SOURCE_SHA), ("sparse_store_root", STORE_ROOT),
                    ("sparse_store_audit", STORE_ROOT + "/audit.json"),
                    ("sparse_store_frames_sha256", FRAMES_SHA)):
                require(data.get(field) == expected_value, f"YAML data.{field} mismatch")
            for field, path in (("sparse_store_audit_sha256", STORE_ROOT + "/audit.json"),
                                ("sparse_store_metadata_sha256", STORE_ROOT + "/store.json"),
                                ("sparse_store_index_sha256", STORE_ROOT + "/index.jsonl")):
                require(data.get(field) == PINS[path][1], f"YAML {field} mismatch")
            fold_record = folds["folds"][fold - 1]
            for name, suffix in (("manifest_sha256", "manifest_sha256"),
                                 ("manifest_samples", "samples"), ("manifest_setups", "setups")):
                expected_value = fold_record[f"{role}_{suffix}"]
                require(record.get(name) == data.get(name) == expected_value, f"Fold field mismatch: {name}")
            require(data["num_frames"] == (8, 16, 32, 64)[VARIANTS.index(variant)], "Temporal budget changed")
            require(cfg["training"]["epochs"] == 65, "Epoch schedule changed")
            expected_argv = ["python", f"scripts/{action}_ksp.py", "--config", expected["config"]]
            if action == "evaluate":
                expected_argv += ["--checkpoint", expected["checkpoint"], "--role", "heldout"]
            expected_argv += ["--device", "cuda"]
            require(argv == expected_argv, "Ordered command/artifact correspondence mismatch")
        require(seen == expected_cartesian, "Missing Cartesian job")
    require(len(configs_seen) == 96 and set(config_bytes) == configs_seen, "Unexpected config inventory")
    wanted = set(itertools.product((1, 2, 3), ("warm_reuse", "fresh_mapping"), range(1, 6)))
    records, commands = plan["artifacts"]["benchmarks"], plan["benchmark_commands"]
    require(len(records) == len(commands) == 30, "Incomplete benchmark matrix")
    seen = set()
    for record, argv in zip(records, commands, strict=True):
        key = record["fold"], record["cache_regime"], record["replicate"]
        require(key in wanted and key not in seen, "Duplicate/unexpected benchmark job")
        seen.add(key)
        fold, regime, rep = key
        output = f"{RUN_ROOT}/stage1/seed_271828/benchmarks/outer_fold_{fold}/{regime}/replicate_{rep}.npz"
        require(record.get("base_seed") == 271828 and record.get("variants") == list(VARIANTS), "Cost identity changed")
        require(record.get("raw_timing_path") == output, "Cost output mismatch")
        expected = ["python", "scripts/benchmark_ksp_system.py", "--protocol-config", PROTOCOL_PATH,
                    "--resolved-root", RESOLVED_ROOT, "--stage", "stage1", "--base-seed", "271828",
                    "--fold", str(fold), "--cache-regime", regime, "--replicate", str(rep),
                    "--variants", *VARIANTS, "--device", "cuda", "--output", output]
        require(argv == expected, "Benchmark argv/artifact mismatch")
    require(seen == wanted, "Incomplete benchmark matrix")
    return {"training_jobs": 48, "evaluation_jobs": 48, "benchmark_jobs": 30,
            "resolved_configs": 96, "exact_argv_artifact_yaml_alignment": True}


def build_portable_metadata(code: dict[str, bytes], cpu: dict[str, bytes]) -> tuple[dict[str, bytes], dict]:
    """Use the accepted source bytes; preserve the historical archive separately."""
    plan = json_value(cpu[PLAN_PATH])
    folds = json_value(cpu[FOLD_PLAN_PATH])
    # Existing checker is reused as an input-contract function, not its full disk prelock.
    check_stage1(plan, folds["folds"], cpu, code)
    check_command_exports(plan, cpu)
    outputs = dict(cpu)
    changes = []
    config_bytes = {}
    original_config_bytes = {}
    for kind in ("trainings", "evaluations"):
        for record in plan["artifacts"][kind]:
            path = relative_path(record["config"])
            original = cpu[path]
            converted, diff = convert_yaml(original)
            original_config_bytes[path] = original
            config_bytes[path] = outputs[path] = converted
            changes.append({"path": path, "old_sha256": digest(original), "new_sha256": digest(converted),
                            "kind": "yaml_paths_only", "changes": diff})
    for path in [FOLD_PLAN_PATH] + [f"{FOLD_ROOT}/outer_fold_{i}/fold.json" for i in (1, 2, 3)]:
        converted, diff = convert_paths(json_value(cpu[path]))
        outputs[path] = json_bytes(converted)
        changes.append({"path": path, "old_sha256": digest(cpu[path]), "new_sha256": digest(outputs[path]),
                        "kind": "json_paths_only", "changes": diff})
    portable = copy.deepcopy(plan)
    command_changes = []
    for _, section, _ in COMMAND_KINDS:
        converted_commands = []
        for number, argv in enumerate(plan[section], 1):
            converted, diff = convert_argv(argv)
            converted_commands.append(converted)
            command_changes.extend({"section": section, "record": number, **d} for d in diff)
        portable[section] = converted_commands
        path = RESOLVED_ROOT + "/" + section + ".jsonl"
        outputs[path] = b"".join(json.dumps(a, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                                 for a in converted_commands)
    # Command vectors have a separate, position-aware path converter.
    without_commands = {k: v for k, v in portable.items() if not k.endswith("_commands")}
    converted, diff = convert_paths(without_commands)
    portable.update(converted)
    old_fold_hash = portable["fold_plan_sha256"]
    portable["fold_plan_sha256"] = digest(outputs[FOLD_PLAN_PATH])
    outputs[PLAN_PATH] = json_bytes(portable)
    changes.append({"path": PLAN_PATH, "old_sha256": digest(cpu[PLAN_PATH]),
                    "new_sha256": digest(outputs[PLAN_PATH]), "kind": "paths_and_fold_plan_dependency_hash",
                    "changes": diff, "dependency_hash_change": {"field": "fold_plan_sha256",
                    "old": old_fold_hash, "new": portable["fold_plan_sha256"]}})
    require(len(command_changes) == 234, "Expected precisely the accepted 234 command path changes")
    alignment = validate_job_alignment(portable, config_bytes, json_value(outputs[FOLD_PLAN_PATH]))
    report = {"schema": "aura-har.ksp-portable-migration.v1", "version": VERSION,
              "protocol_id": PROTOCOL_ID, "source_code_archive_sha256": CODE_SHA,
              "source_cpu_archive_sha256": AUDIT_SHA, "files": changes,
              "command_path_changes": command_changes, "alignment": alignment,
              "scientific_values_changed": False, "historical_files_overwritten": False,
              "gpu_authorized": False, "protocol_lock_created": False,
              "original_resolved_config_sha256": {p: digest(b) for p, b in original_config_bytes.items()},
              "portable_resolved_config_sha256": {p: digest(b) for p, b in config_bytes.items()},
              "unchanged_audit_metadata": {p: digest(outputs[p]) for p in (
                  STORE_ROOT + "/store.json", STORE_ROOT + "/audit.json", STORE_ROOT + "/index.jsonl")}}
    return outputs, report


def make_bound_plan(portable: dict, config_bytes: dict[str, bytes]) -> dict:
    """Candidate job graph with stable IDs; no self-hash or not-yet-existing lock hash."""
    jobs = []
    train_ids = {}
    for kind, section, action in COMMAND_KINDS:
        for record, argv in zip(portable["artifacts"][kind], portable[section], strict=True):
            key = {"action": action, "stage": "stage1", "base_seed": 271828,
                   "fold": record["fold"]}
            if action == "benchmark":
                key.update(cache_regime=record["cache_regime"], replicate=record["replicate"])
            else:
                key.update(variant=record["variant"], stream=record["stream"],
                           role=record["role"], job_seed=record["job_seed"])
            job_id = action + "-" + object_sha(key)[:24]
            job = {"job_id": job_id, "identity": key, "artifact": copy.deepcopy(record),
                   "argv": list(argv), "depends_on": []}
            if action != "benchmark":
                job["config_sha256"] = digest(config_bytes[record["config"]])
                pair = record["fold"], record["variant"], record["stream"]
                if action == "train":
                    train_ids[pair] = job_id
                else:
                    job["depends_on"] = [train_ids[pair]]
            else:
                job["depends_on"] = [train_ids[(record["fold"], v, s)] for v in VARIANTS for s in STREAMS]
            jobs.append(job)
    require(len(jobs) == len({j["job_id"] for j in jobs}) == 126, "Job identity collision")
    return {"schema": "aura-har.ksp-bound-plan-candidate.v1", "protocol_id": PROTOCOL_ID,
            "status": "prepared_not_executable", "executable": False, "gpu_authorized": False,
            "old_validation_used": False, "test_read": False, "external_confirmation": False,
            "primary_candidate": "k32_exact_uniform", "base_seeds": [271828],
            "source_portable_plan": PLAN_PATH, "source_portable_plan_sha256": digest(json_bytes(portable)),
            "jobs": jobs,
            "hash_graph": "configs -> plan -> lock -> authorization/runtime-envelope; no self-hashes"}
