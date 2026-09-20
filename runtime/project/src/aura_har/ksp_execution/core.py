"""Fail-closed binding primitives for the next KSP execution release.

No entry point is unlocked by importing this module. Candidate schemas are NOT
accepted here. Public CLI integration is implemented in the parent package; these
shape-checking functions alone are not an execution permission.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import re
from pathlib import Path
from typing import Any

from .portable import (
    PROTOCOL_ID,
    SETUPS,
    SOURCE_SHA,
    STREAMS,
    TPC_LOCK_SHA,
    VARIANTS,
    object_sha,
    path_under,
    relative_path,
    require,
)

SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT_RULE = "final_epoch_65_no_validation_monitoring"
LOCK_SCHEMA = "aura-har.ksp-protocol-lock.v1"
PLAN_SCHEMA = "aura-har.ksp-bound-plan.v1"
AUTH_SCHEMA = "aura-har.ksp-execution-authorization.v1"
ENVELOPE_SCHEMA = "aura-har.ksp-runtime-envelope.v1"


def checked_sha(value: Any, name: str) -> str:
    require(isinstance(value, str) and SHA_PATTERN.fullmatch(value) is not None,
            f"Invalid SHA-256: {name}")
    require(value not in {"0" * 64, "f" * 64}, f"Placeholder SHA-256: {name}")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock_shape(lock: dict, plan: dict, *, actual_plan_sha256: str) -> None:
    """Pure schema/cross-reference checks. File verification must follow separately."""
    require(isinstance(lock, dict) and isinstance(plan, dict), "Lock and plan must be mappings")
    require(lock.get("schema") == LOCK_SCHEMA, "Candidate/unknown lock cannot authorize execution")
    require(lock.get("status") == "frozen_after_cpu_audit_before_stage1", "Unfrozen protocol lock")
    require(plan.get("schema") == PLAN_SCHEMA, "Candidate/unknown plan cannot authorize execution")
    require(lock.get("protocol_id") == plan.get("protocol_id") == PROTOCOL_ID, "Protocol ID mismatch")
    require(lock.get("plan_sha256") == checked_sha(actual_plan_sha256, "actual_plan_sha256"), "Plan SHA mismatch")
    for field in ("old_validation_used", "test_read", "external_confirmation"):
        require(lock.get(field) is False and plan.get(field) is False, f"Forbidden boundary: {field}")
    source = lock.get("source", {})
    require(source == {"hash_scheme": "canonical-jsonl-v1", "manifest_sha256": SOURCE_SHA,
                       "samples": 38670, "setups": list(SETUPS)}, "Frozen source boundary changed")
    scientific = lock.get("scientific_files")
    require(isinstance(scientific, dict) and scientific, "Empty scientific file binding")
    # A final lock cannot leave its implementation or guarded entry points unbound.
    mandatory = {"configs/ksp/ntu120_xset_trainonly_sparse.yaml", "configs/ksp/ntu120_xset_trainonly_base.yaml",
                 "protocols/AURA_KSP_NTU120_XSET_TRAINONLY_PREREGISTRATION_RU.md",
                 "src/aura_har/ksp_protocol.py", "src/aura_har/ksp_runtime.py",
                 "src/aura_har/ksp_analysis.py", "src/aura_har/data/sparse_store.py",
                 "src/aura_har/data/sparse_reference.py", "scripts/train_ksp.py",
                 "scripts/evaluate_ksp.py", "scripts/benchmark_ksp_system.py",
                 "scripts/stitch_ksp_oof.py", "scripts/analyze_ksp_development.py"}
    require(mandatory <= set(scientific), "Incomplete scientific source binding")
    for path, sha in scientific.items():
        require(relative_path(path) == path, "Scientific paths must already be portable")
        checked_sha(sha, path)
    require(plan.get("base_seeds") == [271828] and plan.get("primary_candidate") == "k32_exact_uniform",
            "This binding core accepts only unchanged Stage 1")
    validate_job_graph(plan)
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 126, "Incomplete Stage-1 job matrix")
    ids = [j.get("job_id") for j in jobs]
    require(all(isinstance(j, str) and j for j in ids) and len(set(ids)) == 126, "Duplicate/invalid job IDs")
    required_config_ids = {j["job_id"] for j in jobs if j["identity"]["action"] in {"train", "evaluate"}}
    config_map = lock.get("resolved_config_sha256", {})
    require(len(required_config_ids) == 96 and set(config_map) == required_config_ids, "Incomplete config bindings")
    for job in jobs:
        require(job["identity"].get("stage") == "stage1" and job["identity"].get("base_seed") == 271828,
                "Wrong job scope")
        require("plan_sha256" not in job and "protocol_lock_sha256" not in job,
                "Serialized jobs must not create a self/circular hash dependency")
        if job["job_id"] in config_map:
            require(config_map[job["job_id"]] == checked_sha(job.get("config_sha256"), "job config"),
                    "Lock/job config mismatch")
        require(set(job.get("depends_on", [])) <= set(ids), "Unknown job dependency")
    fold = lock.get("fold_plan", {})
    checked_sha(fold.get("sha256"), "fold_plan")
    partitions = fold.get("partition_manifest_sha256", {})
    require(set(partitions) == {f"outer_fold_{f}/{r}" for f in (1, 2, 3) for r in ("train", "heldout")},
            "Incomplete partition bindings")
    for key, value in partitions.items():
        checked_sha(value, key)
    store = lock.get("sparse_store", {})
    require(set(store) == {"metadata_sha256", "index_sha256", "frames_sha256", "deep_audit_sha256"},
            "Incomplete store binding")
    for key, value in store.items():
        checked_sha(value, key)
    legacy = lock.get("legacy_tpc", {})
    require(legacy.get("locked_files") == 29, "Closed TPC boundary changed")
    require(legacy.get("lock_sha256") == TPC_LOCK_SHA, "Historical TPC lock identity changed")


def validate_authorization(authorization: dict, *, lock_sha256: str, plan_sha256: str) -> None:
    """Validate a supplied owner authorization; never create one from draft flags."""
    require(isinstance(authorization, dict) and authorization.get("schema") == AUTH_SCHEMA,
            "Missing/invalid separate execution authorization")
    require(authorization.get("protocol_id") == PROTOCOL_ID, "Authorization protocol mismatch")
    require(authorization.get("protocol_lock_sha256") == checked_sha(lock_sha256, "lock"), "Authorization/lock mismatch")
    require(authorization.get("plan_sha256") == checked_sha(plan_sha256, "plan"), "Authorization/plan mismatch")
    require(authorization.get("scope") == "stage1_only", "Only Stage 1 is supported")
    require(authorization.get("gpu_authorized") is True, "Explicit GPU authorization absent")
    require(authorization.get("old_validation_authorized") is False and
            authorization.get("test_read_authorized") is False, "Forbidden data authorization")


def verify_file_map(root: Path, mapping: dict[str, str]) -> None:
    require(isinstance(mapping, dict) and mapping, "Empty file map is not a verification")
    for relative, expected in mapping.items():
        path = path_under(root, relative)
        require(path.is_file(), f"Bound file missing: {relative}")
        require(_sha_file(path) == checked_sha(expected, relative), f"Bound file changed: {relative}")


def runtime_envelope(job: dict, *, lock_sha256: str, plan_sha256: str,
                     store: dict, source_manifest_sha256: str = SOURCE_SHA) -> dict:
    """Attach downstream hashes after the plan/lock were serialized, not inside them."""
    require(job.get("identity", {}).get("stage") == "stage1", "Unknown runtime scope")
    require(source_manifest_sha256 == SOURCE_SHA, "Wrong runtime source")
    for field in ("metadata_sha256", "index_sha256", "frames_sha256", "deep_audit_sha256"):
        checked_sha(store.get(field), field)
    return {"schema": ENVELOPE_SCHEMA, "protocol_id": PROTOCOL_ID,
            "protocol_lock_sha256": checked_sha(lock_sha256, "lock"),
            "plan_sha256": checked_sha(plan_sha256, "plan"),
            "job_id": job["job_id"], "identity": copy.deepcopy(job["identity"]),
            "config_sha256": checked_sha(job.get("config_sha256"), "config"),
            "manifest_sha256": checked_sha(job["artifact"].get("manifest_sha256"), "manifest"),
            "source_manifest_sha256": source_manifest_sha256,
            "sparse_store": copy.deepcopy(store), "checkpoint_rule": CHECKPOINT_RULE}


def validate_checkpoint_binding(payload: dict, expected: dict, *, final: bool) -> None:
    """Inspect metadata in an already deserialized payload, never load a .pt here."""
    require(isinstance(payload, dict), "Checkpoint payload must be a mapping")
    require(payload.get("ksp_binding") == expected, "Checkpoint belongs to a different bound job")
    epoch = payload.get("epoch")
    require(type(epoch) is int and 1 <= epoch <= 65, "Invalid checkpoint epoch")
    require(expected.get("checkpoint_rule") == CHECKPOINT_RULE, "Checkpoint rule changed")
    if final:
        require(epoch == 65, "Evaluation requires epoch-65 final.pt")
    require("model" in payload, "Checkpoint has no model weights")


def validate_job_graph(plan: dict) -> None:
    """Exact Stage-1 identity/dependency matrix; cycles are impossible after this check."""
    jobs = plan.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 126, "Incomplete job graph")
    train_ids = {}
    identity_counts = {a: set() for a in ("train", "evaluate", "benchmark")}
    for job in jobs:
        require(isinstance(job, dict) and isinstance(job.get("identity"), dict), "Invalid job record")
        ident = job["identity"]
        action = ident.get("action")
        require(action in identity_counts, "Unknown job action")
        require(ident.get("stage") == "stage1" and type(ident.get("base_seed")) is int
                and ident["base_seed"] == 271828, "Wrong job seed/stage")
        fold = ident.get("fold")
        require(type(fold) is int and fold in (1, 2, 3), "Wrong job fold")
        if action == "benchmark":
            key = fold, ident.get("cache_regime"), ident.get("replicate")
            require(type(ident.get("replicate")) is int, "Invalid repetition type")
            expected_keys = {"action", "stage", "base_seed", "fold", "cache_regime", "replicate"}
        else:
            key = fold, ident.get("variant"), ident.get("stream")
            require(ident.get("variant") in VARIANTS and ident.get("stream") in STREAMS, "Unknown K/stream")
            seed = 271828 + 1000 * (fold - 1) + 10 * STREAMS.index(ident["stream"])
            require(type(ident.get("job_seed")) is int and ident["job_seed"] == seed, "Unpaired job seed")
            require(ident.get("role") == ("train" if action == "train" else "heldout"), "Job role changed")
            expected_keys = {"action", "stage", "base_seed", "fold", "variant", "stream", "role", "job_seed"}
        require(set(ident) == expected_keys, "Unexpected identity field")
        expected_id = action + "-" + object_sha(ident)[:24]
        require(job.get("job_id") == expected_id, "Nondeterministic or changed job ID")
        require(key not in identity_counts[action], "Duplicate graph job")
        identity_counts[action].add(key)
        artifact = job.get("artifact", {})
        for name in ("base_seed", "fold"):
            require(type(artifact.get(name)) is int and artifact[name] == ident[name], "Artifact identity mismatch")
        for name in (("cache_regime", "replicate") if action == "benchmark" else
                     ("variant", "stream", "role", "job_seed")):
            require(artifact.get(name) == ident[name], "Artifact/identity disagreement")
        if action == "train":
            train_ids[key] = expected_id
    cartesian = set(itertools.product((1, 2, 3), VARIANTS, STREAMS))
    require(identity_counts["train"] == identity_counts["evaluate"] == cartesian, "Missing training/evaluation job")
    require(identity_counts["benchmark"] == set(itertools.product((1, 2, 3),
                ("warm_reuse", "fresh_mapping"), range(1, 6))), "Wrong benchmark matrix")
    for job in jobs:
        ident = job["identity"]
        action, fold = ident["action"], ident["fold"]
        expected = []
        if action == "evaluate":
            expected = [train_ids[fold, ident["variant"], ident["stream"]]]
        elif action == "benchmark":
            expected = [train_ids[fold, variant, stream] for variant in VARIANTS for stream in STREAMS]
        require(job.get("depends_on") == expected, "Changed job dependencies or cycle")
