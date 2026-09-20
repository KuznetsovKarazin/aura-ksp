from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aura_har.data.ntu120 import setup_from_sample_id
from aura_har.ksp_execution import (
    allow_verified_context,
)
from aura_har.utils.io import (
    CANONICAL_JSONL_HASH_SCHEME,
    canonical_jsonl_sha256,
    read_jsonl,
    sha256_file,
)

PROTOCOL_ID = "aura-ksp-ntu120-xset-trainonly-sparse-v1"
METHOD_NAME = "AURA-KSP"
PROTOCOL_CONFIG_SCHEMA = "aura-har.ksp-protocol-config.v1"
RUN_CONFIG_SCHEMA = "aura-har.ksp-run-config.v1"
COMMAND_PLAN_SCHEMA = "aura-har.ksp-command-plan.v1"
FOLD_PLAN_SCHEMA = "aura-har.ksp-fold-plan.v1"
STORE_AUDIT_SCHEMA = "aura-har.ntu120-frame-store-audit.v1"
RESULT_SCHEMA = "aura-har.sparse-temporal-efficiency.v2"
RESULT_VERSION = 2
PROTOCOL_STATUS = "draft_before_data_audit_and_gpu"
DRAFT_LOCK_STATE = "absent_by_design"
STATE_D_ALLOWED_STAGES = (
    "build-sparse-store",
    "data-audit",
    "build-folds",
    "materialize-stage1",
    "materialize-conditional-repeatability",
    "dry-run",
    "regression-tests",
    "ruff",
)
STATE_D_BLOCKED_SCIENTIFIC_STAGES = (
    "train",
    "evaluate",
    "benchmark",
    "stitch",
    "analyze",
)
PLANNED_SCIENTIFIC_STAGES = (
    "oof-train",
    "oof-evaluate",
    "cost-audit",
    "stitch-oof",
    "analyze",
    "conditional-repeatability",
)

TRAIN_MANIFEST_SHA256 = "fd037a9f71dbb00778bf71428892c8de407577402ee8330fdba0b02cea20edaa"
TRAIN_SAMPLES = 38_670
DEVELOPMENT_SETUPS = (4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32)
OUTER_SETUP_FOLDS = ((4, 12, 18, 26), (6, 14, 22, 28), (8, 16, 24, 32))
OUTER_FOLD_COUNTS = (
    {"train_samples": 26_105, "heldout_samples": 12_565},
    {"train_samples": 26_144, "heldout_samples": 12_526},
    {"train_samples": 25_091, "heldout_samples": 13_579},
)
OLD_VALIDATION_SHA256 = "2ece0ff18ea434cea3815dfeaf4832cb527d2f64dbbc5167f7389d9d2892ff35"
OLD_VALIDATION_SAMPLES = 15_798
OLD_VALIDATION_SETUPS = (2, 10, 20, 30)
SEALED_TEST_SHA256 = "5aed365468ad753ffd6887df88109841155f302db6cc033459c79cd9afd3194f"
SEALED_TEST_SAMPLES = 59_477

STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
FIXED_PRIOR = (0.3, 0.3, 0.2, 0.2)
PRIMARY_VARIANT_ID = "k32_exact_uniform"
PRIMARY_SEED = 271_828
CONDITIONAL_SEEDS = (161_803, 141_421)
FORBIDDEN_LEGACY_SEEDS = (42, 123, 2026)
STAGE1_TRAININGS = 48
STAGE1_EVALUATIONS = 48
CONDITIONAL_TRAININGS = 48
CONDITIONAL_EVALUATIONS = 48

ALL_CLASS_TOP1_MARGIN = 0.010
NEWCLASSES_TOP1_MARGIN = 0.015
MACRO_F1_MARGIN = 0.015
MINIMUM_COST_SAVING = 0.20
FAMILY_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_DRAW_MATRIX_SHA256 = (
    "bcb73c25f714fa77017682e866a5ce7de4ba74d7a34dd940abc9e3dce172f7da"
)
TPC_LOCK_SHA256 = "74796b903bfc6f702541b3a841a74648baaf4d34c2de2a8a7e01798ad66ed305"
TPC_LOCKED_FILES = 29


@dataclass(frozen=True)
class KSPVariant:
    id: str
    budget: int
    temporal_mode: str
    sampler: str
    role: str
    align_corners: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


VARIANTS = (
    KSPVariant("k8_exact_uniform", 8, "exact_k", "uniform", "secondary_candidate"),
    KSPVariant("k16_exact_uniform", 16, "exact_k", "uniform", "secondary_candidate"),
    KSPVariant("k32_exact_uniform", 32, "exact_k", "uniform", "primary_candidate"),
    KSPVariant(
        "k64_resize_reference",
        64,
        "resize",
        "bilinear",
        "primary_operational_reference",
        False,
    ),
)


def expected_variants() -> list[dict[str, Any]]:
    return [variant.to_dict() for variant in VARIANTS]


def variant_by_id(variant_id: str) -> KSPVariant:
    for variant in VARIANTS:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"Unknown AURA-KSP variant: {variant_id}")


def job_seed(base_seed: int, fold_index: int, stream: str) -> int:
    """Pair initialization and shuffle across K within a fold and stream."""
    if not 1 <= int(fold_index) <= len(OUTER_SETUP_FOLDS):
        raise ValueError(f"Outer fold must be 1..{len(OUTER_SETUP_FOLDS)}")
    try:
        stream_index = STREAMS.index(str(stream))
    except ValueError as error:
        raise ValueError(f"Unknown AURA-KSP stream: {stream}") from error
    return int(base_seed) + 1000 * (int(fold_index) - 1) + 10 * stream_index


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"Frozen AURA-KSP field changed: {name}={actual!r}, expected {expected!r}")


def require_ksp_execution_authorization(
    config: Mapping[str, Any], *, stage: str
) -> None:
    """Fail closed while this release intentionally has no KSP lock/authorization."""
    if stage not in STATE_D_BLOCKED_SCIENTIFIC_STAGES:
        raise ValueError(f"Unknown AURA-KSP scientific execution stage: {stage}")
    if config.get("schema") not in {PROTOCOL_CONFIG_SCHEMA, RUN_CONFIG_SCHEMA}:
        raise ValueError("AURA-KSP execution guard received an unsupported config schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("AURA-KSP execution guard requires a protocol mapping")
    _require_equal(protocol.get("id"), PROTOCOL_ID, "execution protocol")
    _require_equal(protocol.get("status"), PROTOCOL_STATUS, "execution protocol status")
    _require_equal(protocol.get("lock_state"), DRAFT_LOCK_STATE, "execution lock state")
    _require_equal(
        protocol.get("execution_authorized"), False, "execution authorization state"
    )
    _require_equal(protocol.get("gpu_authorized"), False, "execution GPU authorization")
    _require_equal(protocol.get("external_confirmation"), False, "execution claim boundary")
    if allow_verified_context(config, stage=stage):
        return
    raise PermissionError(
        "AURA-KSP STATE D is draft_unlocked_nonexecutable: "
        f"{stage} is blocked on CPU and GPU until a real protocol lock and a separate "
        "explicit-user execution authorization are released"
    )


def validate_protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if a draft config changes the pre-data-audit scientific design."""
    _require_equal(config.get("schema"), PROTOCOL_CONFIG_SCHEMA, "schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    _require_equal(protocol.get("id"), PROTOCOL_ID, "protocol.id")
    _require_equal(protocol.get("method"), METHOD_NAME, "protocol.method")
    _require_equal(protocol.get("status"), PROTOCOL_STATUS, "protocol.status")
    _require_equal(protocol.get("lock_state"), DRAFT_LOCK_STATE, "protocol.lock_state")
    _require_equal(
        protocol.get("execution_authorized"), False, "protocol.execution_authorized"
    )
    _require_equal(protocol.get("gpu_authorized"), False, "protocol.gpu_authorized")
    _require_equal(protocol.get("external_confirmation"), False, "external_confirmation")
    _require_equal(
        tuple(protocol.get("allowed_stages", [])),
        STATE_D_ALLOWED_STAGES,
        "STATE D allowed stages",
    )
    _require_equal(
        tuple(protocol.get("planned_scientific_stages_not_authorized", [])),
        PLANNED_SCIENTIFIC_STAGES,
        "STATE D planned scientific stages",
    )
    forbidden_stages = set(protocol.get("forbidden_stages", []))
    required_forbidden = {"old-validation", "final-train", "final-test", "sealed-test"}
    if not required_forbidden.issubset(forbidden_stages):
        raise ValueError("All old-validation/final/test stages must be forbidden")

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    source = data.get("source_manifest")
    if not isinstance(source, Mapping):
        raise TypeError("data.source_manifest must be a mapping")
    _require_equal(source.get("sha256"), TRAIN_MANIFEST_SHA256, "train manifest SHA-256")
    _require_equal(source.get("samples"), TRAIN_SAMPLES, "train samples")
    _require_equal(tuple(source.get("setups", [])), DEVELOPMENT_SETUPS, "train setups")
    _require_equal(tuple(tuple(fold) for fold in data.get("outer_folds", [])), OUTER_SETUP_FOLDS, "folds")
    _require_equal(data.get("group_unit"), "setup", "group unit")

    excluded = data.get("excluded_partitions")
    if not isinstance(excluded, Mapping):
        raise TypeError("excluded partition ledger is required")
    old_validation = excluded.get("old_validation")
    sealed_test = excluded.get("xset_test")
    if not isinstance(old_validation, Mapping) or not isinstance(sealed_test, Mapping):
        raise TypeError("Both excluded partition ledger entries are required")
    _require_equal(old_validation.get("manifest_sha256"), OLD_VALIDATION_SHA256, "old val hash")
    _require_equal(old_validation.get("used_by_protocol"), False, "old val use")
    _require_equal(sealed_test.get("manifest_sha256"), SEALED_TEST_SHA256, "test hash")
    _require_equal(sealed_test.get("read_by_protocol"), False, "test read")
    if "path" in old_validation or "path" in sealed_test:
        raise ValueError("Excluded partition metadata must not contain payload paths")

    sparse = config.get("sparse_store")
    if not isinstance(sparse, Mapping):
        raise TypeError("sparse_store must be a mapping")
    _require_equal(sparse.get("schema"), "aura-har.ntu120-frame-store.v1", "store schema")
    _require_equal(sparse.get("frame_file"), "frames.npy", "store frame file")
    _require_equal(sparse.get("index_file"), "index.jsonl", "store index file")
    _require_equal(sparse.get("frame_layout"), "T,C,V,M", "store frame layout")
    _require_equal(tuple(sparse.get("frame_shape_cvm", [])), (3, 25, 2), "store frame shape")
    _require_equal(sparse.get("shared_source_read_across_streams"), True, "shared source read")
    _require_equal(
        sparse.get("exact_k_read_rule"),
        "selected_unique_source_indices_only",
        "exact-K read rule",
    )
    _require_equal(sparse.get("resize64_align_corners"), False, "resize64 align_corners")
    _require_equal(
        sparse.get("resize64_read_rule"),
        "exact_bilinear_interpolation_support_only",
        "resize64 read rule",
    )
    _require_equal(sparse.get("resize64_max_unique_support_frames"), 128, "resize64 support")

    _require_equal(config.get("variants"), expected_variants(), "variant family")
    fusion = config.get("fusion")
    if not isinstance(fusion, Mapping):
        raise TypeError("fusion must be a mapping")
    _require_equal(fusion.get("name"), "fixed_prior", "fusion comparator")
    _require_equal(tuple(fusion.get("streams", [])), STREAMS, "stream order")
    _require_equal(tuple(float(value) for value in fusion.get("weights", [])), FIXED_PRIOR, "prior")
    _require_equal(fusion.get("quality_reference"), "k64_resize_reference", "K64 reference")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TypeError("training must be a mapping")
    _require_equal(training.get("primary_seed"), PRIMARY_SEED, "primary seed")
    _require_equal(tuple(training.get("conditional_seeds", [])), CONDITIONAL_SEEDS, "conditional seeds")
    _require_equal(
        tuple(training.get("forbidden_legacy_seeds", [])),
        FORBIDDEN_LEGACY_SEEDS,
        "forbidden legacy seeds",
    )
    _require_equal(training.get("stage1_trainings"), STAGE1_TRAININGS, "stage-1 trainings")
    _require_equal(training.get("stage1_evaluations"), STAGE1_EVALUATIONS, "stage-1 evaluations")
    _require_equal(training.get("conditional_trainings"), CONDITIONAL_TRAININGS, "conditional trainings")
    _require_equal(
        training.get("conditional_evaluations"),
        CONDITIONAL_EVALUATIONS,
        "conditional evaluations",
    )
    _require_equal(
        tuple(training.get("conditional_variants", [])),
        (PRIMARY_VARIANT_ID, "k64_resize_reference"),
        "conditional variants",
    )
    _require_equal(training.get("conditional_requires_primary_gate_pass"), True, "conditional gate")
    _require_equal(training.get("epochs"), 65, "training epochs")
    _require_equal(training.get("checkpoint_rule"), "final_epoch_65_no_validation_monitoring", "checkpoint")
    _require_equal(training.get("validate_every"), 0, "validation monitoring")
    _require_equal(training.get("primary_candidate"), PRIMARY_VARIANT_ID, "primary candidate")

    statistics = config.get("statistics")
    gate = config.get("development_gate")
    if not isinstance(statistics, Mapping) or not isinstance(gate, Mapping):
        raise TypeError("statistics and development_gate mappings are required")
    _require_equal(statistics.get("family_alpha"), FAMILY_ALPHA, "family alpha")
    _require_equal(statistics.get("bootstrap_resamples"), BOOTSTRAP_RESAMPLES, "bootstrap")
    _require_equal(statistics.get("seed"), PRIMARY_SEED, "bootstrap seed")
    _require_equal(statistics.get("cluster_unit"), "setup", "cluster unit")
    _require_equal(statistics.get("primary_candidate"), PRIMARY_VARIANT_ID, "primary hypothesis")
    _require_equal(
        statistics.get("multiplicity"),
        "holm_over_k8_k16_k32_composite_tests",
        "multiplicity",
    )
    _require_equal(
        statistics.get("secondary_candidates_cannot_rescue_primary"),
        True,
        "secondary rescue rule",
    )
    _require_equal(
        statistics.get("bootstrap_method"),
        "setup_cluster_resample_with_replacement_centered_basic",
        "bootstrap method",
    )
    _require_equal(
        statistics.get("shared_draw_matrix"),
        "across_all_k_endpoints_and_cache_regimes",
        "bootstrap draw sharing",
    )
    _require_equal(
        tuple(statistics.get("draw_matrix_shape", [])),
        (BOOTSTRAP_RESAMPLES, len(DEVELOPMENT_SETUPS)),
        "bootstrap draw matrix shape",
    )
    _require_equal(
        statistics.get("draw_matrix_sha256"),
        BOOTSTRAP_DRAW_MATRIX_SHA256,
        "bootstrap draw matrix SHA-256",
    )
    _require_equal(statistics.get("quantile_method"), "higher", "bootstrap quantile")
    _require_equal(
        statistics.get("margin_null_p_value"),
        "centered_one_sided_tail_plus_one",
        "margin-null p-value",
    )
    _require_equal(
        statistics.get("cost_cache_regime_iut"),
        "min_lcb_and_max_p",
        "cost cache-regime IUT",
    )
    _require_equal(
        statistics.get("percentile_tail_reinterpretation_allowed"),
        False,
        "percentile-tail reinterpretation",
    )
    _require_equal(
        statistics.get("exact_distribution_free_claim"),
        False,
        "exact distribution-free claim",
    )
    _require_equal(
        statistics.get("inference_scope"),
        "setup_cluster_uncertainty_not_joint_hardware_run_uncertainty",
        "inference scope",
    )
    _require_equal(gate.get("all_class_top1_lower_strictly_greater_than"), -ALL_CLASS_TOP1_MARGIN, "Top-1 gate")
    _require_equal(gate.get("newclasses_top1_lower_strictly_greater_than"), -NEWCLASSES_TOP1_MARGIN, "new-class gate")
    _require_equal(gate.get("macro_f1_lower_strictly_greater_than"), -MACRO_F1_MARGIN, "macro-F1 gate")
    _require_equal(gate.get("end_to_end_cost_saving_lower_at_least"), MINIMUM_COST_SAVING, "cost gate")
    _require_equal(gate.get("primary_composite_p_below"), FAMILY_ALPHA, "primary p gate")
    _require_equal(gate.get("primary_p_value_role"), "holm_adjusted_over_all_three_k", "p role")
    _require_equal(gate.get("failure_consequence"), "report_negative_without_retuning", "failure rule")
    _require_equal(gate.get("pass_consequence"), "allow_conditional_repeatability_only", "pass rule")

    cost = config.get("cost")
    if not isinstance(cost, Mapping):
        raise TypeError("cost must be a mapping")
    _require_equal(cost.get("schema"), "aura-har.ksp-system-cost-audit.v2", "cost schema")
    _require_equal(
        cost.get("boundary"),
        "shared_sparse_read_four_stream_batch1_same_device",
        "cost boundary",
    )
    _require_equal(tuple(cost.get("cache_regimes", [])), ("warm_reuse", "fresh_mapping"), "cache")
    _require_equal(cost.get("batch_size"), 1, "cost batch size")
    _require_equal(cost.get("num_workers"), 0, "cost workers")
    _require_equal(cost.get("warmup"), 50, "cost warmup")
    _require_equal(cost.get("warmup_scope"), "per_variant", "cost warmup scope")
    _require_equal(cost.get("samples_per_setup"), 150, "cost sample count")
    _require_equal(cost.get("process_repetitions"), 5, "cost repetitions")
    _require_equal(cost.get("raw_paired_timings_required"), True, "raw cost traces")
    _require_equal(cost.get("aggregate_only_audit_rejected"), True, "aggregate cost guard")

    abstention = config.get("abstention")
    if not isinstance(abstention, Mapping):
        raise TypeError("abstention must be a mapping")
    _require_equal(
        abstention.get("enabled"), False, "v1 abstention enabled state"
    )
    _require_equal(
        abstention.get("implemented_in_v1"), False, "v1 abstention implementation"
    )
    _require_equal(
        abstention.get("role"),
        "future_extension_requires_new_lock_and_protocol_lineage",
        "abstention role",
    )
    _require_equal(
        abstention.get("can_rescue_static_failure"), False, "abstention rescue rule"
    )
    _require_equal(
        abstention.get("prerequisite"),
        "static_k32_pass_and_actual_incremental_cost_benchmark",
        "abstention extension prerequisite",
    )
    _require_equal(
        abstention.get("claim_allowed_in_v1"), False, "v1 abstention claim"
    )

    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping")
    _require_equal(outputs.get("result_schema"), RESULT_SCHEMA, "output schema")
    _require_equal(outputs.get("artifact_version"), RESULT_VERSION, "output version")
    _require_equal(outputs.get("test_read"), False, "output test status")
    return dict(config)


def validate_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_samples: int = TRAIN_SAMPLES,
    expected_setups: Sequence[int] = DEVELOPMENT_SETUPS,
) -> tuple[str, ...]:
    """Validate semantic train-only provenance after the exact source hash check."""
    if len(rows) != int(expected_samples):
        raise ValueError(f"Unexpected source sample count: {len(rows)} != {expected_samples}")
    allowed = tuple(sorted(int(value) for value in expected_setups))
    sample_ids: list[str] = []
    observed_setups: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Manifest rows must be mappings")
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Manifest row misses sample_id")
        parsed_setup = setup_from_sample_id(sample_id)
        row_setup = int(row.get("setup", parsed_setup))
        if row_setup != parsed_setup:
            raise ValueError(f"Manifest setup/sample ID mismatch: {sample_id}")
        if row_setup not in allowed:
            raise ValueError(f"Non-development setup in KSP source: {row_setup}")
        label = int(row.get("label", -1))
        if not 0 <= label < 120:
            raise ValueError(f"Invalid NTU120 label in KSP source: {label}")
        sample_ids.append(sample_id)
        observed_setups.add(row_setup)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in KSP source")
    if tuple(sorted(observed_setups)) != allowed:
        raise ValueError("KSP source does not contain the exact frozen setup set")
    return tuple(sample_ids)


def holm_adjusted_pvalues(pvalues: Sequence[float]) -> list[float]:
    values = [float(value) for value in pvalues]
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Holm p-values must be finite values in [0, 1]")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def evaluate_primary_gate(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen K32 intersection-union development gate."""
    required = (
        "all_class_top1_lower",
        "newclasses_top1_lower",
        "macro_f1_lower",
        "end_to_end_cost_saving_lower",
        "primary_holm_adjusted_composite_p",
    )
    values: dict[str, float] = {}
    for key in required:
        try:
            value = float(assessment[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Primary KSP assessment misses a numeric {key}") from error
        if not math.isfinite(value):
            raise ValueError(f"Primary KSP assessment has non-finite {key}")
        values[key] = value
    components = {
        "all_class_top1": values["all_class_top1_lower"] > -ALL_CLASS_TOP1_MARGIN,
        "newclasses_top1": values["newclasses_top1_lower"] > -NEWCLASSES_TOP1_MARGIN,
        "macro_f1": values["macro_f1_lower"] > -MACRO_F1_MARGIN,
        "end_to_end_cost": values["end_to_end_cost_saving_lower"] >= MINIMUM_COST_SAVING,
        "primary_holm_adjusted_composite_p": (
            values["primary_holm_adjusted_composite_p"] < FAMILY_ALPHA
        ),
    }
    return {
        "primary_candidate": PRIMARY_VARIANT_ID,
        "components": components,
        "gate_passed": all(components.values()),
        "failure_consequence": "conditional_seeds_blocked_and_report_negative_without_retuning",
    }


def validate_result_provenance(result: Mapping[str, Any]) -> None:
    _require_equal(result.get("schema"), RESULT_SCHEMA, "result schema")
    _require_equal(result.get("artifact_version"), RESULT_VERSION, "artifact version")
    _require_equal(result.get("protocol_id"), PROTOCOL_ID, "result protocol")
    _require_equal(result.get("method"), METHOD_NAME, "result method")
    _require_equal(
        result.get("source_role"),
        "ntu120_xset_train_only_nested_setup_oof",
        "result source role",
    )
    _require_equal(result.get("selection_baseline"), "fixed_prior", "result comparator")
    _require_equal(tuple(result.get("prior", [])), FIXED_PRIOR, "result prior")
    _require_equal(result.get("primary_candidate"), PRIMARY_VARIANT_ID, "result primary")
    _require_equal(result.get("old_validation_used"), False, "result old validation use")
    _require_equal(result.get("test_read"), False, "result test read")
    _require_equal(result.get("external_confirmation"), False, "result external claim")


def _forbidden_command_token(value: str) -> bool:
    normalized = str(value).replace("\\", "/").lower()
    forbidden = (
        "/test/",
        "eval_test",
        "final_test",
        "final-test",
        "test_manifest",
        "manifests/test.jsonl",
        "manifests/val.jsonl",
        "/old_validation/",
        "/sealed",
    )
    return any(token in normalized for token in forbidden)


def validate_command_argv(command: Sequence[str]) -> None:
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("AURA-KSP commands must be non-empty argv arrays")
    if any(_forbidden_command_token(value) for value in command):
        raise ValueError("AURA-KSP development command addresses forbidden validation/test data")
    for index, value in enumerate(command[:-1]):
        if value in {"--split", "--role"} and command[index + 1].lower() not in {
            "train",
            "heldout",
            "oof",
        }:
            raise ValueError("AURA-KSP accepts only train-derived roles")


def validate_command_plan(plan: Mapping[str, Any]) -> None:
    _require_equal(plan.get("schema"), COMMAND_PLAN_SCHEMA, "plan schema")
    _require_equal(plan.get("protocol_id"), PROTOCOL_ID, "plan protocol")
    _require_equal(plan.get("gpu_authorized"), False, "plan GPU authorization")
    _require_equal(plan.get("old_validation_used"), False, "plan old validation")
    _require_equal(plan.get("test_read"), False, "plan test read")
    _require_equal(plan.get("external_confirmation"), False, "plan external claim")
    for field in (
        "sparse_store_audit_sha256",
        "sparse_store_metadata_sha256",
        "sparse_store_index_sha256",
        "sparse_store_frames_sha256",
    ):
        _validate_sha256(plan.get(field), f"plan {field}")
    _validate_sha256(plan.get("fold_plan_sha256"), "plan fold_plan_sha256")
    if not str(plan.get("fold_plan", "")):
        raise ValueError("AURA-KSP command plan requires the validated fold plan path")
    if not str(plan.get("sparse_store_audit", "")):
        raise ValueError("AURA-KSP command plan requires the passed deep store audit path")
    mode = plan.get("mode")
    if mode == "stage1_train_only":
        expected = {
            "train_commands": STAGE1_TRAININGS,
            "evaluation_commands": STAGE1_EVALUATIONS,
            "benchmark_commands": 30,
        }
        _require_equal(plan.get("base_seeds"), [PRIMARY_SEED], "stage-1 seeds")
    elif mode == "conditional_repeatability":
        expected = {
            "train_commands": CONDITIONAL_TRAININGS,
            "evaluation_commands": CONDITIONAL_EVALUATIONS,
            "benchmark_commands": 60,
        }
        _require_equal(plan.get("base_seeds"), list(CONDITIONAL_SEEDS), "conditional seeds")
        _require_equal(plan.get("requires_primary_gate_pass"), True, "conditional gate")
        _require_equal(plan.get("primary_candidate"), PRIMARY_VARIANT_ID, "conditional K")
    else:
        raise ValueError(f"Unsupported AURA-KSP command-plan mode: {mode}")
    for section, count in expected.items():
        commands = plan.get(section)
        if not isinstance(commands, list) or len(commands) != count:
            raise ValueError(f"AURA-KSP {section} count must be {count}")
        for command in commands:
            validate_command_argv(command)
    artifacts = plan.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, Mapping):
            raise TypeError("AURA-KSP plan artifacts must be a mapping")
        artifact_sections = {
            "trainings": expected["train_commands"],
            "evaluations": expected["evaluation_commands"],
            "benchmarks": expected["benchmark_commands"],
        }
        for section, count in artifact_sections.items():
            records = artifacts.get(section)
            if not isinstance(records, list) or len(records) != count:
                raise ValueError(f"AURA-KSP artifact {section} count must be {count}")
            for record in records:
                if not isinstance(record, Mapping):
                    raise TypeError("AURA-KSP artifact records must be mappings")
                if any(
                    _forbidden_command_token(str(value))
                    for value in record.values()
                    if isinstance(value, (str, Path))
                ):
                    raise ValueError("AURA-KSP artifact record addresses forbidden data")
                if section in {"trainings", "evaluations"}:
                    role = "train" if section == "trainings" else "heldout"
                    _require_equal(
                        int(record.get("fold", -1)),
                        int(record.get("outer_fold", -2)),
                        "artifact fold alias",
                    )
                    validate_fold_partition_fields(record, role=role)
    if any(seed in set(plan.get("base_seeds", [])) for seed in FORBIDDEN_LEGACY_SEEDS):
        raise ValueError("Legacy SPR/TPC seeds are forbidden in the new KSP plan")


def reject_manifest_path_before_read(path: str | Path) -> None:
    """First guard only; the exact hash and semantic rows remain mandatory."""
    normalized = Path(path).as_posix().lower()
    name = Path(path).name.lower()
    if name != "train.jsonl" or any(token in normalized for token in ("/test/", "/val/", "odd")):
        raise ValueError("KSP fold builder accepts only the canonical train.jsonl path")


def expected_fold_partition(outer_fold: int, role: str) -> dict[str, Any]:
    """Return the frozen setup/count contract for one train-derived outer-fold role."""
    fold = int(outer_fold)
    if not 1 <= fold <= len(OUTER_SETUP_FOLDS):
        raise ValueError(f"AURA-KSP outer_fold must be 1..{len(OUTER_SETUP_FOLDS)}")
    normalized_role = "heldout" if str(role) == "val" else str(role)
    if normalized_role not in {"train", "heldout"}:
        raise ValueError("AURA-KSP fold role must be train or heldout")
    heldout_setups = tuple(int(value) for value in OUTER_SETUP_FOLDS[fold - 1])
    train_setups = tuple(
        setup for setup in DEVELOPMENT_SETUPS if setup not in set(heldout_setups)
    )
    counts = OUTER_FOLD_COUNTS[fold - 1]
    return {
        "outer_fold": fold,
        "role": normalized_role,
        "manifest_samples": int(counts[f"{normalized_role}_samples"]),
        "manifest_setups": train_setups if normalized_role == "train" else heldout_setups,
        "manifest_name": "train.jsonl" if normalized_role == "train" else "heldout.jsonl",
    }


def _validate_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"AURA-KSP {name} must be a lowercase SHA-256")
    return text


def validate_fold_partition_fields(
    value: Mapping[str, Any], *, role: str | None = None
) -> dict[str, Any]:
    """Validate materialized fold identity without opening its manifest payload."""
    actual_role = str(value.get("role", ""))
    if role is not None:
        normalized_role = "heldout" if str(role) == "val" else str(role)
        _require_equal(actual_role, normalized_role, "run data role")
    expected = expected_fold_partition(int(value.get("outer_fold", -1)), actual_role)
    _require_equal(int(value.get("manifest_samples", -1)), expected["manifest_samples"], "fold samples")
    _require_equal(
        tuple(int(item) for item in value.get("manifest_setups", [])),
        expected["manifest_setups"],
        "fold setups",
    )
    manifest_hash = _validate_sha256(value.get("manifest_sha256"), "manifest_sha256")
    manifest = str(value.get("manifest", "")).replace("\\", "/")
    if not manifest or Path(manifest).name != expected["manifest_name"]:
        raise ValueError(
            f"AURA-KSP {actual_role} run requires {expected['manifest_name']}: {manifest}"
        )
    expected_parent = f"outer_fold_{expected['outer_fold']}"
    if Path(manifest).parent.name != expected_parent:
        raise ValueError(
            f"AURA-KSP manifest must belong to {expected_parent}: {manifest}"
        )
    return {**expected, "manifest": manifest, "manifest_sha256": manifest_hash}


def validate_fold_plan(plan: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Validate the complete CPU fold plan before any run config is materialized."""
    _require_equal(plan.get("schema"), FOLD_PLAN_SCHEMA, "fold plan schema")
    _require_equal(plan.get("protocol_id"), PROTOCOL_ID, "fold plan protocol")
    _require_equal(
        plan.get("manifest_hash_scheme"),
        CANONICAL_JSONL_HASH_SCHEME,
        "fold manifest hash scheme",
    )
    _require_equal(
        plan.get("source_manifest_sha256"), TRAIN_MANIFEST_SHA256, "fold source hash"
    )
    _require_equal(int(plan.get("source_samples", -1)), TRAIN_SAMPLES, "fold source samples")
    _require_equal(
        tuple(int(value) for value in plan.get("source_setups", [])),
        DEVELOPMENT_SETUPS,
        "fold source setups",
    )
    _require_equal(plan.get("heldout_coverage"), "exactly_once", "fold coverage")
    _require_equal(plan.get("old_validation_used"), False, "fold old validation")
    _require_equal(plan.get("test_read"), False, "fold test read")
    _require_equal(plan.get("external_confirmation"), False, "fold external claim")
    reject_manifest_path_before_read(str(plan.get("source_manifest", "")))
    folds = plan.get("folds")
    if not isinstance(folds, list) or len(folds) != len(OUTER_SETUP_FOLDS):
        raise ValueError("AURA-KSP fold plan must contain exactly three folds")
    resolved: dict[int, dict[str, Any]] = {}
    for raw in folds:
        if not isinstance(raw, Mapping):
            raise TypeError("AURA-KSP fold records must be mappings")
        fold = int(raw.get("fold", -1))
        if fold in resolved:
            raise ValueError(f"Duplicate AURA-KSP fold record: {fold}")
        train_value = {
            "outer_fold": fold,
            "role": "train",
            "manifest": raw.get("train_manifest"),
            "manifest_sha256": raw.get("train_manifest_sha256"),
            "manifest_samples": raw.get("train_samples"),
            "manifest_setups": raw.get("train_setups"),
        }
        heldout_value = {
            "outer_fold": fold,
            "role": "heldout",
            "manifest": raw.get("heldout_manifest"),
            "manifest_sha256": raw.get("heldout_manifest_sha256"),
            "manifest_samples": raw.get("heldout_samples"),
            "manifest_setups": raw.get("heldout_setups"),
        }
        resolved[fold] = {
            "fold": fold,
            "train": validate_fold_partition_fields(train_value, role="train"),
            "heldout": validate_fold_partition_fields(heldout_value, role="heldout"),
        }
    if tuple(sorted(resolved)) != tuple(range(1, len(OUTER_SETUP_FOLDS) + 1)):
        raise ValueError("AURA-KSP fold plan IDs must be exactly 1,2,3")
    return resolved


def validate_run_config(config: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    """Validate a materialized train or train-derived held-out run before any I/O."""
    _require_equal(config.get("schema"), RUN_CONFIG_SCHEMA, "run schema")
    protocol = config.get("protocol")
    data = config.get("data")
    training = config.get("training")
    if not isinstance(protocol, Mapping) or not isinstance(data, Mapping):
        raise TypeError("Materialized KSP protocol/data mappings are required")
    _require_equal(protocol.get("id"), PROTOCOL_ID, "run protocol")
    _require_equal(protocol.get("status"), PROTOCOL_STATUS, "run protocol status")
    _require_equal(protocol.get("lock_state"), DRAFT_LOCK_STATE, "run lock state")
    _require_equal(
        protocol.get("execution_authorized"), False, "run execution authorization"
    )
    _require_equal(protocol.get("gpu_authorized"), False, "run GPU authorization")
    _require_equal(protocol.get("external_confirmation"), False, "run external claim")
    allowed_roles = {"train": "train", "heldout": "heldout", "val": "heldout"}
    if split not in allowed_roles:
        raise ValueError("AURA-KSP exposes only train and train-derived heldout roles")
    _require_equal(data.get("role"), allowed_roles[split], "run data role")
    _require_equal(data.get("source_manifest_sha256"), TRAIN_MANIFEST_SHA256, "source hash")
    manifest = str(data.get("manifest", "")).replace("\\", "/").lower()
    if not manifest or any(
        token in manifest
        for token in ("manifests/val.jsonl", "manifests/test.jsonl", "old_validation", "sealed")
    ):
        raise ValueError("AURA-KSP run addresses a forbidden manifest")
    validate_fold_partition_fields(data, role=allowed_roles[split])
    store = str(data.get("sparse_store_root", "")).replace("\\", "/").lower()
    if not store or any(token in store for token in ("/test/", "old_validation", "sealed")):
        raise ValueError("AURA-KSP run addresses a forbidden sparse store")
    audit_path = str(data.get("sparse_store_audit", "")).replace("\\", "/").lower()
    if not audit_path or any(
        token in audit_path for token in ("/test/", "old_validation", "sealed")
    ):
        raise ValueError("AURA-KSP run addresses a forbidden sparse-store audit")
    for field in (
        "sparse_store_audit_sha256",
        "sparse_store_metadata_sha256",
        "sparse_store_index_sha256",
        "sparse_store_frames_sha256",
    ):
        _validate_sha256(data.get(field), field)
    variant = variant_by_id(str(data.get("variant_id")))
    _require_equal(int(data.get("num_frames", -1)), variant.budget, "run temporal budget")
    _require_equal(data.get("temporal_mode"), variant.temporal_mode, "run temporal mode")
    _require_equal(data.get("sampler"), variant.sampler, "run sampler")
    if variant.align_corners is not None:
        _require_equal(data.get("align_corners"), variant.align_corners, "align_corners")
    _require_equal(data.get("modality") in STREAMS, True, "run modality")
    sampling_seed = data.get("sampling_seed")
    if not isinstance(sampling_seed, int) or sampling_seed <= 0:
        raise ValueError("Materialized KSP sampling_seed must be a positive integer")
    if not isinstance(training, Mapping):
        raise TypeError("Materialized KSP training mapping is required")
    _require_equal(training.get("epochs"), 65, "run epochs")
    _require_equal(training.get("validate_every"), 0, "run validation monitoring")
    _require_equal(
        training.get("checkpoint_rule"),
        "final_epoch_65_no_validation_monitoring",
        "run checkpoint rule",
    )
    return dict(config)


def _resolve_store_member(root: Path, value: Any, *, name: str) -> Path:
    relative = Path(str(value).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"AURA-KSP sparse-store {name} path must be relative")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"AURA-KSP sparse-store {name} path escapes its root") from error
    return target


def validate_deep_store_audit(
    audit: Mapping[str, Any],
    *,
    store_root: str | Path,
    project_root: str | Path | None = None,
    verify_frames_sha256: bool = True,
) -> dict[str, Any]:
    """Validate a frozen full-source audit and, by default, its frame payload hash."""
    _require_equal(audit.get("schema"), STORE_AUDIT_SCHEMA, "store audit schema")
    _require_equal(audit.get("status"), "passed", "store audit status")
    _require_equal(audit.get("deep"), True, "store audit depth")
    _require_equal(
        audit.get("manifest_hash_scheme"),
        CANONICAL_JSONL_HASH_SCHEME,
        "store audit manifest hash scheme",
    )
    _require_equal(audit.get("source_manifest_sha256"), TRAIN_MANIFEST_SHA256, "audit source hash")
    _require_equal(int(audit.get("samples", -1)), TRAIN_SAMPLES, "audit source samples")
    _require_equal(
        tuple(int(value) for value in audit.get("setups", [])),
        DEVELOPMENT_SETUPS,
        "audit source setups",
    )
    _require_equal(audit.get("gpu_used"), False, "store audit GPU use")
    _require_equal(audit.get("old_validation_used"), False, "store audit old validation")
    _require_equal(audit.get("test_read"), False, "store audit test read")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping):
        raise TypeError("AURA-KSP store audit checks must be a mapping")
    required_checks = {
        "frames_sha256",
        "semantic_content_sha256",
        "source_frame_equality",
        "source_manifest_sha256",
        "source_manifest_metadata",
    }
    if not required_checks.issubset(
        {str(key) for key, value in checks.items() if value is True}
    ):
        raise ValueError("AURA-KSP store audit lacks full source-equivalence checks")

    root = Path(store_root).expanduser().resolve()
    metadata_path = root if root.is_file() else root / "store.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"AURA-KSP sparse-store metadata is required: {metadata_path}")
    actual_metadata_sha256 = sha256_file(metadata_path)
    expected_metadata_sha256 = _validate_sha256(
        audit.get("store_metadata_sha256"), "audit store_metadata_sha256"
    )
    _require_equal(actual_metadata_sha256, expected_metadata_sha256, "store metadata hash")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise TypeError("AURA-KSP sparse-store metadata must be a JSON object")
    _require_equal(metadata.get("schema"), "aura-har.ntu120-frame-store.v1", "store schema")
    _require_equal(
        metadata.get("source_manifest_sha256"), TRAIN_MANIFEST_SHA256, "store source hash"
    )
    _require_equal(int(metadata.get("samples", -1)), TRAIN_SAMPLES, "store samples")
    _require_equal(
        tuple(int(value) for value in metadata.get("setups", [])),
        DEVELOPMENT_SETUPS,
        "store setups",
    )
    _require_equal(metadata.get("test_read"), False, "store test read")
    expected_index_sha256 = _validate_sha256(
        audit.get("store_index_sha256"), "audit store_index_sha256"
    )
    expected_frames_sha256 = _validate_sha256(
        audit.get("store_frames_sha256"), "audit store_frames_sha256"
    )
    _require_equal(metadata.get("index_sha256"), expected_index_sha256, "store index hash")
    _require_equal(metadata.get("frames_sha256"), expected_frames_sha256, "store frames hash")
    index_path = _resolve_store_member(
        metadata_path.parent, metadata.get("index_path"), name="index"
    )
    _require_equal(sha256_file(index_path), expected_index_sha256, "actual store index hash")
    frames_path = _resolve_store_member(
        metadata_path.parent, metadata.get("frames_path"), name="frames"
    )
    if not frames_path.is_file():
        raise FileNotFoundError(f"AURA-KSP sparse-store frame payload is required: {frames_path}")
    if verify_frames_sha256:
        _require_equal(
            sha256_file(frames_path),
            expected_frames_sha256,
            "actual store frames hash",
        )

    resolved_project_root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    actual_boundary = verify_legacy_tpc_boundary(resolved_project_root)
    boundary = audit.get("legacy_tpc_boundary")
    if not isinstance(boundary, Mapping):
        raise TypeError("AURA-KSP store audit requires the legacy TPC boundary record")
    _require_equal(dict(boundary), actual_boundary, "store audit TPC boundary")
    return {
        "source_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "samples": TRAIN_SAMPLES,
        "setups": list(DEVELOPMENT_SETUPS),
        "sparse_store_metadata_sha256": actual_metadata_sha256,
        "sparse_store_index_sha256": expected_index_sha256,
        "sparse_store_frames_sha256": expected_frames_sha256,
        "metadata_path": str(metadata_path),
        "validated_deep_audit": True,
        "frames_payload_sha256_verified": bool(verify_frames_sha256),
        "test_read": False,
    }


def validate_run_store_provenance(
    config: Mapping[str, Any], *, project_root: str | Path | None = None
) -> dict[str, Any]:
    """Verify the content-addressed deep store audit before dataset construction."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("Materialized KSP data mapping is required")
    audit_path = Path(str(data.get("sparse_store_audit", "")))
    if not audit_path.is_file():
        raise FileNotFoundError(f"AURA-KSP sparse-store audit is required: {audit_path}")
    actual_audit_sha256 = sha256_file(audit_path)
    _require_equal(
        actual_audit_sha256,
        _validate_sha256(data.get("sparse_store_audit_sha256"), "sparse_store_audit_sha256"),
        "sparse-store audit hash",
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise TypeError("AURA-KSP sparse-store audit must be a JSON object")
    resolved = validate_deep_store_audit(
        audit,
        store_root=str(data["sparse_store_root"]),
        project_root=project_root,
        verify_frames_sha256=False,
    )
    for field in (
        "sparse_store_metadata_sha256",
        "sparse_store_index_sha256",
        "sparse_store_frames_sha256",
    ):
        _require_equal(data.get(field), resolved[field], f"run {field}")
    return {
        **resolved,
        "sparse_store_audit": str(audit_path),
        "sparse_store_audit_sha256": actual_audit_sha256,
        "validated_before_dataset_io": True,
    }


def validate_run_manifest_provenance(
    config: Mapping[str, Any], *, split: str
) -> dict[str, Any]:
    """Verify exact fold hash/count/setups before a sparse store or dataset is opened."""
    validate_run_config(config, split=split)
    store_provenance = validate_run_store_provenance(config)
    data = config["data"]
    role = "heldout" if split == "val" else split
    expected = validate_fold_partition_fields(data, role=role)
    manifest = Path(str(data["manifest"]))
    actual_hash = canonical_jsonl_sha256(manifest)
    if actual_hash != expected["manifest_sha256"]:
        raise ValueError(
            "AURA-KSP canonical fold manifest SHA-256 mismatch: "
            f"{actual_hash} != {expected['manifest_sha256']}"
        )
    rows = read_jsonl(manifest)
    validate_source_rows(
        rows,
        expected_samples=int(expected["manifest_samples"]),
        expected_setups=expected["manifest_setups"],
    )
    return {
        **store_provenance,
        "outer_fold": expected["outer_fold"],
        "role": role,
        "manifest": str(manifest),
        "manifest_sha256": actual_hash,
        "manifest_samples": len(rows),
        "manifest_setups": list(expected["manifest_setups"]),
        "validated_before_dataset_io": True,
        "test_read": False,
    }


def verify_legacy_tpc_boundary(project_root: str | Path) -> dict[str, Any]:
    """Prove that the closed TPC lock and all 29 locked files remain byte-identical."""
    root = Path(project_root).resolve()
    lock_path = root / "protocols/AURA_TPC_NTU120_NEWCLASSES_XSET_LOCK.json"
    expected_lock = TPC_LOCK_SHA256
    actual_lock = sha256_file(lock_path)
    if actual_lock != expected_lock:
        raise ValueError(f"Closed TPC lock changed: {actual_lock} != {expected_lock}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    files = lock.get("files", {})
    if len(files) != TPC_LOCKED_FILES:
        raise ValueError(
            f"Closed TPC lock must contain {TPC_LOCKED_FILES} files, got {len(files)}"
        )
    mismatches = []
    for relative, expected in files.items():
        target = root / relative
        actual = sha256_file(target) if target.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"Closed TPC boundary changed: {mismatches[:3]}")
    return {
        "lock_sha256": actual_lock,
        "locked_files": TPC_LOCKED_FILES,
        "mismatches": 0,
    }
