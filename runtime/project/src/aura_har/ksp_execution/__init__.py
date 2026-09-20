"""Stage-1 KSP execution binding. No consent or execution occurs at import.

The five public CLIs establish a verified context before their original guard.
Unbound library calls retain the original synthetic-test behavior; their outputs
cannot be consumed by a bound CLI because bound receipts are mandatory.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from . import core, portable
from .pins import (
    FRAMES_SHA,
    PINS,
    SOURCE_PATH,
    STORE_ROOT,
    TPC_LOCK_PATH,
    TPC_LOCK_SHA,
    canonical_hash,
    json_rows,
    json_value,
    sha_file,
)

VERSION = "1.0.1"
LOCK_PATH = "execution/KSP_STAGE1_LOCK.json"
BOUND_PLAN_PATH = "execution/KSP_STAGE1_PLAN.json"
RECEIPT_SCHEMA = "aura-har.ksp-completion-receipt.v1"
FILE_SCHEMA = "aura-har.ksp-bound-file.v1"
FRAME_RECEIPT_PATH = "execution/HOST_FRAME_VERIFICATION.json"
FRAME_FILE_BYTES = 1681636328
SOURCE_PIN = "92efb974d18c6a5eb8b30f4161c81d6dde4378a62e323a83b219f97cf5e5ad22"
PREPARED_PIN = "7ddb1b4c80bf4b59fd9999afe4464faad7369824bbb2450eabd426bf2960cbf3"
OOF_ROOT = portable.RUN_ROOT + "/oof_artifacts"
COST_ROOT = portable.RUN_ROOT + "/stage1/seed_271828/benchmarks"
ANALYSIS_ROOT = "reports/ksp_trainonly/stage1"
_ACTIVE = ContextVar("ksp_verified_execution", default=None)


def require(value, message):
    if not value:
        raise PermissionError("AURA-KSP execution binding: " + message)


def read_json(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32 * 1024 * 1024,
            "Missing/oversize JSON: " + str(path))
    return json_value(path.read_bytes())


def atomic_json(path, value):
    """Replace a receipt only after the corresponding payload has been saved."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(portable.json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def make_post_jobs(plan):
    evaluations = [j["job_id"] for j in plan["jobs"] if j["identity"]["action"] == "evaluate"]
    benchmarks = [j["job_id"] for j in plan["jobs"] if j["identity"]["action"] == "benchmark"]
    result = []
    for action in ("stitch", "analyze"):
        ident = {"action": action, "stage": "stage1", "base_seed": 271828}
        jid = action + "-" + portable.object_sha(ident)[:24]
        if action == "stitch":
            argv = ["python", "scripts/stitch_ksp_oof.py", "--plan", portable.PLAN_PATH,
                    "--protocol-config", portable.PROTOCOL_PATH, "--manifest", SOURCE_PATH,
                    "--output-dir", OOF_ROOT]
            outputs = [f"{OOF_ROOT}/seed_271828/{v}.npz" for v in portable.VARIANTS]
            outputs.append(OOF_ROOT + "/stitch_report.json")
            deps = evaluations
        else:
            argv = ["python", "scripts/analyze_ksp_development.py", "--protocol-config", portable.PROTOCOL_PATH,
                    "--oof-root", OOF_ROOT, "--cost-root", COST_ROOT,
                    "--store-audit", STORE_ROOT + "/audit.json", "--output-dir", ANALYSIS_ROOT]
            outputs = [ANALYSIS_ROOT + "/development_summary.json", ANALYSIS_ROOT + "/gate_table.csv"]
            deps = [result[0]["job_id"], *benchmarks]
        result.append({"job_id": jid, "identity": ident, "argv": argv,
                       "depends_on": deps, "output_files": outputs})
    return result


def required_outputs(job):
    action = job["identity"]["action"]
    record = job.get("artifact", {})
    if action == "train":
        target = record["checkpoint"]
        return [target, str(Path(target).parent.as_posix()) + "/training_summary.json"]
    if action == "evaluate":
        target = record["predictions_path"]
        directory = str(Path(target).parent.as_posix())
        return [target, directory + "/evaluation_provenance.json", directory + "/metrics.json"]
    if action == "benchmark":
        target = record["raw_timing_path"]
        return [target, str(Path(target).with_suffix(".json").as_posix())]
    return list(job["output_files"])


def validate_final_records(lock, plan, plan_sha):
    core.validate_lock_shape(lock, plan, actual_plan_sha256=plan_sha)
    require(plan.get("status") == "locked_awaiting_separate_authorization", "Plan is not frozen")
    require(plan.get("executable") is True and plan.get("gpu_authorized") is False,
            "Plan capability is not an owner authorization")
    require(lock.get("execution_implementation") == VERSION, "Unsupported implementation")
    require(lock.get("upstream_code_archive_sha256") == SOURCE_PIN, "Wrong upstream release")
    require(lock.get("prepared_archive_sha256") == PREPARED_PIN, "Wrong accepted CPU build")
    require(plan.get("postprocessing_jobs") == make_post_jobs(plan), "Postprocessing graph changed")
    require(plan.get("source_portable_plan") == portable.PLAN_PATH, "Unknown underlying plan")
    require(plan.get("source_portable_plan_sha256") == lock.get("portable_plan_sha256"),
            "Portable plan binding changed")
    required_paths = {"src/aura_har/ksp_execution/" + name for name in
                      ("__init__.py", "core.py", "portable.py", "pins.py")}
    required_paths.update({"scripts/run_ksp_bound.py", "scripts/verify_ksp_host.py"})
    require(required_paths <= set(lock["scientific_files"]), "Execution implementation not locked")
    required_data = {SOURCE_PATH, portable.FOLD_PLAN_PATH,
                     STORE_ROOT + "/store.json", STORE_ROOT + "/audit.json", STORE_ROOT + "/index.jsonl"}
    required_data |= {f"{portable.FOLD_ROOT}/outer_fold_{f}/{role}.jsonl"
                     for f in (1, 2, 3) for role in ("train", "heldout")}
    require(set(lock.get("metadata_files", {})) == required_data, "Incomplete metadata binding")
    for field, name in (("metadata_sha256", "store.json"), ("index_sha256", "index.jsonl"),
                        ("deep_audit_sha256", "audit.json")):
        expected = PINS[STORE_ROOT + "/" + name][1]
        require(lock["sparse_store"][field] == expected, "Store pin differs from accepted data")
        require(lock["metadata_files"][STORE_ROOT + "/" + name] == expected, "Metadata/store mismatch")
    require(lock["sparse_store"]["frames_sha256"] == FRAMES_SHA, "Frame payload pin changed")
    require(lock["metadata_files"][portable.FOLD_PLAN_PATH] == lock["fold_plan"]["sha256"],
            "Fold-plan binding mismatch")


class ExecutionContext:
    def __init__(self, root, lock, plan, lock_sha, plan_sha, auth_sha):
        self.root = Path(root).resolve()
        self.lock, self.plan = lock, plan
        self.lock_sha, self.plan_sha, self.auth_sha = lock_sha, plan_sha, auth_sha
        self.jobs = {j["job_id"]: j for j in plan["jobs"] + plan["postprocessing_jobs"]}
        self.job = None
        self.verified_receipts = {}
        self.input_receipts = {}
        self.verified_payloads = {}
        self.expected_config = None

    def path(self, relative):
        return portable.path_under(self.root, relative)

    def relative(self, value):
        value = Path(value)
        actual = value.resolve() if value.is_absolute() else (self.root / value).resolve()
        require(actual.is_relative_to(self.root), "Path escapes project")
        return portable.relative_path(actual.relative_to(self.root).as_posix())

    def binding(self, job=None):
        job = self.job if job is None else job
        require(job is not None, "No selected job")
        if job["identity"]["action"] in {"train", "evaluate"}:
            return core.runtime_envelope(job, lock_sha256=self.lock_sha, plan_sha256=self.plan_sha,
                                         store=self.lock["sparse_store"])
        return {"schema": core.ENVELOPE_SCHEMA, "protocol_id": portable.PROTOCOL_ID,
                "protocol_lock_sha256": self.lock_sha, "plan_sha256": self.plan_sha,
                "job_id": job["job_id"], "identity": copy.deepcopy(job["identity"]),
                "source_manifest_sha256": portable.SOURCE_SHA,
                "sparse_store": copy.deepcopy(self.lock["sparse_store"])}

    def receipt_path(self, job):
        return self.path("execution/receipts/" + job["job_id"] + ".json")

    def verify_completion(self, job_id):
        if job_id in self.verified_receipts:
            return self.verified_receipts[job_id]
        job = self.jobs[job_id]
        receipt_path = self.receipt_path(job)
        receipt = read_json(receipt_path)
        require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("status") == "complete",
                "Incomplete job receipt: " + job_id)
        require(receipt.get("binding") == self.binding(job), "Mixed-run completion receipt")
        outputs = receipt.get("outputs", {})
        require(set(outputs) == set(required_outputs(job)), "Unexpected completion output set")
        core.verify_file_map(self.root, outputs)
        expected_inputs = {dependency: self.verify_completion(dependency) for dependency in job["depends_on"]}
        require(receipt.get("input_receipt_sha256") == expected_inputs, "Dependency receipt changed")
        for name in outputs:
            if name.endswith((".pt", ".npz")):
                self.verify_file_receipt(name, job)
        digest = sha_file(receipt_path)
        self.verified_receipts[job_id] = digest
        return digest

    def verify_file_receipt(self, relative, job):
        path = self.path(relative)
        value = read_json(Path(str(path) + ".ksp.json"))
        require(value.get("schema") == FILE_SCHEMA and value.get("path") == relative,
                "Missing/misplaced bound-file receipt")
        require(value.get("binding") == self.binding(job), "Wrong file job identity")
        require(value.get("sha256") == sha_file(path) and value.get("bytes") == path.stat().st_size,
                "Payload changed since receipt")
        return value

    def checkpoint_job(self, path):
        relative = self.relative(path)
        candidates = []
        for job in self.jobs.values():
            if job["identity"]["action"] != "train":
                continue
            final = job["artifact"]["checkpoint"]
            if relative == final or (self.job is job and
                    Path(relative).parent == Path(final).parent and Path(relative).name == "last.pt"):
                candidates.append(job)
        require(len(candidates) == 1, "Checkpoint is not a frozen training output")
        job = candidates[0]
        require(job is self.job or job["job_id"] in self.job["depends_on"], "Checkpoint outside job dependencies")
        return relative, job

    def load_checkpoint(self, path):
        import torch
        relative, job = self.checkpoint_job(path)
        self.verify_file_receipt(relative, job)
        if relative in self.verified_payloads:
            return self.verified_payloads[relative]
        with self.path(relative).open("rb") as handle:
            before = sha_file(self.path(relative))
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        require(before == sha_file(self.path(relative)), "Checkpoint changed during deserialization")
        final = job is not self.job or Path(relative).name == "final.pt"
        core.validate_checkpoint_binding(payload, self.binding(job), final=final)
        if job is self.job:
            require(1 <= payload["epoch"] < 65, "Resume requires unfinished epoch 1..64")
            require(all(key in payload for key in ("optimizer", "scheduler", "rng_state")),
                    "Incomplete resume state")
        self.verified_payloads[relative] = payload
        return payload

    def output_allowed(self, path):
        relative = self.relative(path)
        action = self.job["identity"]["action"]
        roots = {str(Path(p).parent.as_posix()) for p in required_outputs(self.job)}
        require(str(Path(relative).parent.as_posix()) in roots, "Output outside selected job")
        if action == "train" and relative.endswith(".pt"):
            require(Path(relative).name in {"last.pt", "final.pt"} or
                    re.fullmatch(r"epoch_[0-9]{3}\.pt", Path(relative).name) is not None,
                    "Unfrozen checkpoint name")
        return relative

    def file_receipt(self, path):
        relative = self.output_allowed(path)
        target = self.path(relative)
        atomic_json(str(target) + ".ksp.json", {"schema": FILE_SCHEMA, "path": relative,
                    "binding": self.binding(), "sha256": sha_file(target), "bytes": target.stat().st_size})


def inspect_lock(root, lock_path=LOCK_PATH):
    root = Path(root).resolve()
    lock_file = portable.path_under(root, lock_path)
    lock = read_json(lock_file)
    plan_file = portable.path_under(root, BOUND_PLAN_PATH)
    plan = read_json(plan_file)
    plan_sha, lock_sha = sha_file(plan_file), sha_file(lock_file)
    validate_final_records(lock, plan, plan_sha)
    core.verify_file_map(root, lock["scientific_files"])
    source_files = {p.relative_to(root).as_posix() for p in (root / "src/aura_har").rglob("*.py")}
    require(source_files <= set(lock["scientific_files"]), "Unbound Python module in source tree")
    require(sha_file(root / portable.PLAN_PATH) == plan["source_portable_plan_sha256"], "Underlying plan changed")
    require(sha_file(root / TPC_LOCK_PATH) == TPC_LOCK_SHA, "Closed TPC lock changed")
    legacy = read_json(root / TPC_LOCK_PATH)
    require(len(legacy.get("files", {})) == 29, "Closed TPC file inventory changed")
    core.verify_file_map(root, legacy["files"])
    config_files = {j["artifact"]["config"]: j["config_sha256"] for j in plan["jobs"] if "config_sha256" in j}
    core.verify_file_map(root, config_files)
    underlying = read_json(root / portable.PLAN_PATH)
    folds = read_json(root / portable.FOLD_PLAN_PATH)
    raw_configs = {p: (root / p).read_bytes() for p in config_files}
    portable.validate_job_alignment(underlying, raw_configs, folds)
    expected = portable.make_bound_plan(underlying, raw_configs)
    require(plan["jobs"] == expected["jobs"], "Bound graph and original matrix differ")
    for f in folds["folds"]:
        for role in ("train", "heldout"):
            key = f"outer_fold_{f['fold']}/{role}"
            require(lock["fold_plan"]["partition_manifest_sha256"][key] == f[role + "_manifest_sha256"],
                    "Partition canonical hash mismatch")
    return ExecutionContext(root, lock, plan, lock_sha, plan_sha, None)


def frame_stat(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino}


def record_host_frames(context):
    """One explicit deployment check, never called inside scientific/timed CLIs."""
    path = context.path(STORE_ROOT + "/frames.npy")
    require(path.is_file() and path.stat().st_size == FRAME_FILE_BYTES,
            "Frame array missing or wrong size after transfer")
    before = frame_stat(path)
    digest = sha_file(path)
    after = frame_stat(path)
    require(before == after and digest == FRAMES_SHA, "Transferred frames changed or have the wrong SHA-256")
    record = {"schema": "aura-har.ksp-host-frame-verification.v1",
              "protocol_id": portable.PROTOCOL_ID, "project_root": str(context.root),
              "protocol_lock_sha256": context.lock_sha, "plan_sha256": context.plan_sha,
              "frame_relative_path": STORE_ROOT + "/frames.npy", "frames_sha256": digest,
              "local_file_stat": after, "verified_utc": datetime.now(timezone.utc).isoformat(),
              "scientific_training_run": False, "gpu_authorized": False, "test_read": False}
    target = context.path(FRAME_RECEIPT_PATH)
    require(not target.exists(), "Host verification already exists; verify it or preserve stale evidence first")
    atomic_json(target, record)
    return record


def verify_host_frames(context):
    """Verify the local deployment receipt using stat only; never warm cache by a full read."""
    record = read_json(context.path(FRAME_RECEIPT_PATH))
    expected = {"schema": "aura-har.ksp-host-frame-verification.v1",
                "protocol_id": portable.PROTOCOL_ID, "project_root": str(context.root),
                "protocol_lock_sha256": context.lock_sha, "plan_sha256": context.plan_sha,
                "frame_relative_path": STORE_ROOT + "/frames.npy", "frames_sha256": FRAMES_SHA}
    require(all(record.get(k) == v for k, v in expected.items()), "Wrong host frame verification binding")
    path = context.path(STORE_ROOT + "/frames.npy")
    require(path.is_file(), "Previously verified frame array is missing")
    actual = frame_stat(path)
    require(actual["bytes"] == FRAME_FILE_BYTES and actual == record.get("local_file_stat"),
            "Frame file changed or moved after host verification; do not execute")
    return record


def verify_authorization(context, auth_path):
    value = read_json(auth_path)
    core.validate_authorization(value, lock_sha256=context.lock_sha, plan_sha256=context.plan_sha)
    require(value.get("explicit_user_consent") is True, "Explicit user consent is absent")
    require(value.get("decision") == "AUTHORIZE_KSP_STAGE1_ONLY", "Unknown owner decision")
    require(isinstance(value.get("authorized_by"), str) and value["authorized_by"].strip(), "Missing owner")
    timestamp = datetime.fromisoformat(value.get("time_utc", ""))
    require(timestamp.tzinfo is not None, "Authorization timestamp requires a timezone")
    context.auth_sha = sha_file(Path(auth_path))


def add_ksp_arguments(parser):
    parser.add_argument("--ksp-lock", help="Frozen project-relative Stage-1 lock")
    parser.add_argument("--ksp-authorization", help="Separate explicit owner authorization JSON")


def select_cli_job(plan, argv, action):
    """Only declared argv is executable. Consent arguments are not scientific overrides."""
    cleaned, options = [], {}
    tokens = list(argv)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--ksp-lock", "--ksp-authorization", "--resume"}:
            require(token not in options and index + 1 < len(tokens), "Duplicate/incomplete execution argument")
            options[token] = tokens[index + 1]
            index += 2
        else:
            cleaned.append(token)
            index += 1
    require(action == "train" or "--resume" not in options, "Resume only applies to training")
    jobs = plan["jobs"] + plan["postprocessing_jobs"]
    matches = [j for j in jobs if j["identity"]["action"] == action and j["argv"][2:] == cleaned]
    require(len(matches) == 1, "CLI does not exactly match a declared Stage-1 command")
    return matches[0], options


def prepare_ksp_cli(args, *, action):
    _ACTIVE.set(None)
    lock_name = getattr(args, "ksp_lock", None)
    auth_name = getattr(args, "ksp_authorization", None)
    if lock_name is None and auth_name is None:
        return  # The original STATE D guard must reject this call.
    require(lock_name is not None and auth_name is not None, "Both lock and authorization are required")
    root = Path.cwd().resolve()
    require(Path(__file__).resolve().is_relative_to(root / "src/aura_har"), "Wrong editable-install source imported")
    for name, module in list(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if name == "aura_har" or name.startswith("aura_har."):
            require(source is None or Path(source).resolve().is_relative_to(root / "src/aura_har"),
                    "Mixed AURA package roots")
    context = inspect_lock(root, portable.relative_path(lock_name))
    # No data payload is opened without the separate authorization.
    verify_authorization(context, Path(auth_name))
    job, options = select_cli_job(context.plan, sys.argv[1:], action)
    context.job = job
    config_path = job.get("artifact", {}).get("config", portable.PROTOCOL_PATH)
    from aura_har.config import load_config
    context.expected_config = copy.deepcopy(load_config(root / config_path))
    context.input_receipts = {dep: context.verify_completion(dep) for dep in job["depends_on"]}
    if action in {"evaluate", "benchmark"}:
        for dependency in job["depends_on"]:
            context.load_checkpoint(context.path(context.jobs[dependency]["artifact"]["checkpoint"]))
        context.verified_payloads.clear()  # Do not hold 16 checkpoints while building GPU models.
    if "--resume" in options:
        context.load_checkpoint(options["--resume"])
    elif any(context.path(name).exists() for name in required_outputs(job)):
        raise FileExistsError("Existing output: use verified completion/explicit resume; overwrite is disabled")
    if action == "train" and "--resume" not in options:
        directory = context.path(job["artifact"]["checkpoint"]).parent
        require(not directory.exists() or not any(directory.iterdir()), "Nonempty training directory without resume")
    core.verify_file_map(root, context.lock["metadata_files"])
    require(canonical_hash(json_rows((root / SOURCE_PATH).read_bytes())) == portable.SOURCE_SHA,
            "Source canonical manifest mismatch")
    if action in {"train", "evaluate", "benchmark"}:
        verify_host_frames(context)
    _ACTIVE.set(context)


def allow_verified_context(config, *, stage):
    context = _ACTIVE.get()
    if context is None:
        return False
    require(context.job["identity"]["action"] == stage, "Runtime stage mismatch")
    def content(value):
        return {k: v for k, v in value.items() if k != "_config_path"}
    require(content(config) == content(context.expected_config), "Config differs from the bound file")
    return True


def atomic_torch_save(payload, path):
    from aura_har.utils.io import atomic_torch_save as original
    context = _ACTIVE.get()
    if context is None:
        return original(payload, path)
    require(context.job["identity"]["action"] == "train", "Checkpoint write outside training")
    relative = context.output_allowed(path)
    bound = dict(payload)
    require("ksp_binding" not in bound, "Upstream payload already has a binding")
    bound["ksp_binding"] = context.binding()
    core.validate_checkpoint_binding(bound, context.binding(), final=Path(relative).name == "final.pt")
    name = Path(relative).name
    if name.startswith("epoch_"):
        epoch = int(name[6:9])
        interval = context.expected_config.get("training", {}).get("save_every", 5)
        require(type(interval) is int and interval > 0, "Invalid frozen checkpoint interval")
        require(epoch == bound["epoch"] and epoch % interval == 0,
                "Periodic checkpoint filename/epoch/interval mismatch")
    original(bound, path)
    context.file_receipt(path)


def load_checkpoint_on_cpu(path):
    context = _ACTIVE.get()
    if context is not None:
        return context.load_checkpoint(path)
    from aura_har.utils.reproducibility import load_checkpoint_on_cpu as original
    return original(path)


def ksp_torch_load(path, *args, **kwargs):
    context = _ACTIVE.get()
    if context is not None:
        return context.load_checkpoint(path)
    import torch
    return torch.load(path, *args, **kwargs)


def load_model_checkpoint(config, checkpoint, device):
    context = _ACTIVE.get()
    if context is None:
        from aura_har.runtime import load_model_checkpoint as original
        return original(config, checkpoint, device)
    from aura_har.models import build_model
    payload = context.load_checkpoint(checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def ksp_savez_compressed(path, *args, **kwargs):
    import numpy as np
    context = _ACTIVE.get()
    if context is not None:
        context.output_allowed(path)
        require("ksp_binding_json" not in kwargs, "Duplicate artifact binding")
        kwargs["ksp_binding_json"] = np.asarray(json.dumps(context.binding(), sort_keys=True, allow_nan=False))
    np.savez_compressed(path, *args, **kwargs)
    if context is not None:
        context.file_receipt(path)


def write_json(value, path):
    from aura_har.utils.io import write_json as original
    context = _ACTIVE.get()
    if context is not None:
        context.output_allowed(path)
        require(isinstance(value, dict) and "ksp_binding" not in value, "Unexpected JSON payload")
        value = {**value, "ksp_binding": context.binding()}
    original(value, path)
    if context is not None:
        context.file_receipt(path)


def finish_ksp_cli():
    context = _ACTIVE.get()
    if context is None:
        return
    job = context.job
    outputs = {name: sha_file(context.path(name)) for name in required_outputs(job)}
    if job["identity"]["action"] == "train":
        context.verify_file_receipt(job["artifact"]["checkpoint"], job)
        summary = read_json(context.path(required_outputs(job)[1]))
        require(summary.get("epochs_completed") == summary.get("epochs_configured") == 65,
                "Training did not finish the configured 65 epochs")
        require(summary.get("validate_every") == 0, "Validation monitoring is forbidden")
    receipt = {"schema": RECEIPT_SCHEMA, "status": "complete", "binding": context.binding(),
               "authorization_sha256": context.auth_sha, "outputs": outputs,
               "input_receipt_sha256": context.input_receipts,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "test_read": False, "old_validation_used": False}
    target = context.receipt_path(job)
    require(not target.exists(), "Completion receipt already exists")
    atomic_json(target, receipt)
    _ACTIVE.set(None)
