from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from aura_har.adaptive_fusion import fuse_logits, normalized_prior

ARTIFACT_SCHEMA = "aura-har.sparse-temporal-efficiency.v2"
PROTOCOL_ID = "aura-ksp-ntu120-xset-trainonly-sparse-v1"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
FIXED_PRIOR = (0.3, 0.3, 0.2, 0.2)
PRIMARY_VARIANT = "k32_exact_uniform"
REFERENCE_VARIANT = "k64_resize_reference"
VARIANTS = ("k8_exact_uniform", "k16_exact_uniform", PRIMARY_VARIANT)


def align_and_fuse_streams(
    archives: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Align four stream artifacts and apply the immutable fixed-prior fusion."""
    if tuple(archives) != STREAMS:
        raise ValueError(f"Expected ordered streams {STREAMS}, got {tuple(archives)}")
    reference = archives[STREAMS[0]]
    labels = np.asarray(reference["labels"], dtype=np.int64)
    sample_ids = np.asarray(reference["sample_ids"]).astype(str)
    setups = np.asarray(reference["setups"], dtype=np.int64)
    stacked = []
    for stream in STREAMS:
        archive = archives[stream]
        logits = np.asarray(archive["logits"], dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] != len(labels):
            raise ValueError(f"Invalid logits for {stream}: {logits.shape}")
        if np.any(~np.isfinite(logits)):
            raise ValueError(f"Non-finite logits for {stream}")
        if not np.array_equal(np.asarray(archive["labels"], dtype=np.int64), labels):
            raise ValueError(f"Label mismatch for {stream}")
        if not np.array_equal(np.asarray(archive["sample_ids"]).astype(str), sample_ids):
            raise ValueError(f"Sample ordering mismatch for {stream}")
        if not np.array_equal(np.asarray(archive["setups"], dtype=np.int64), setups):
            raise ValueError(f"Setup ordering mismatch for {stream}")
        stacked.append(logits)
    stream_logits = np.stack(stacked, axis=1)
    fused = fuse_logits(stream_logits, normalized_prior(FIXED_PRIOR))
    return {
        "logits": np.asarray(fused, dtype=np.float32),
        "labels": labels,
        "sample_ids": sample_ids,
        "setups": setups,
        "stream_logits": stream_logits,
    }


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm input must contain finite probabilities")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        [(len(values) - rank) * values[index] for rank, index in enumerate(order)]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _metric_values(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float]:
    all_top1 = float(np.mean(predictions == labels))
    new_mask = labels >= 60
    if not new_mask.any():
        raise ValueError("A61-A120 endpoint has no samples")
    new_top1 = float(np.mean(predictions[new_mask] == labels[new_mask]))
    macro_f1 = float(
        f1_score(labels, predictions, labels=np.arange(120), average="macro", zero_division=0)
    )
    return all_top1, new_top1, macro_f1


def _cluster_draw_indices(groups: np.ndarray, draws: np.ndarray) -> np.ndarray:
    unique = np.unique(groups)
    lookup = [np.flatnonzero(groups == group) for group in unique]
    return np.concatenate([lookup[int(index)] for index in draws])


def setup_cluster_draws(
    setups: np.ndarray, *, resamples: int, seed: int
) -> dict[str, Any]:
    """Create the one shared frozen setup-resampling matrix and its portable hash."""
    groups = np.asarray(setups, dtype=np.int64)
    unique = np.unique(groups)
    if len(unique) != 12:
        raise ValueError(f"Expected the 12 frozen development setups, got {unique.tolist()}")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(unique),
        size=(int(resamples), len(unique)),
        endpoint=False,
        dtype=np.int64,
    )
    portable = np.ascontiguousarray(draws, dtype="<i8")
    return {
        "groups": unique,
        "draws": draws,
        "resamples": int(resamples),
        "seed": int(seed),
        "draw_matrix_sha256": hashlib.sha256(portable.tobytes()).hexdigest(),
    }


def _validated_draws(
    groups: np.ndarray,
    *,
    resamples: int,
    seed: int,
    draws: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    unique = np.unique(groups)
    if draws is None:
        plan = setup_cluster_draws(groups, resamples=resamples, seed=seed)
        return np.asarray(plan["draws"], dtype=np.int64), str(plan["draw_matrix_sha256"])
    matrix = np.asarray(draws, dtype=np.int64)
    if matrix.shape != (int(resamples), len(unique)):
        raise ValueError("Frozen setup bootstrap draw matrix has the wrong shape")
    if np.any(matrix < 0) or np.any(matrix >= len(unique)):
        raise ValueError("Frozen setup bootstrap draw matrix contains an invalid group index")
    portable = np.ascontiguousarray(matrix, dtype="<i8")
    return matrix, hashlib.sha256(portable.tobytes()).hexdigest()


def paired_accuracy_bootstrap(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    reference_predictions: np.ndarray,
    setups: np.ndarray,
    *,
    resamples: int,
    seed: int,
    draws: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recompute all three accuracy endpoints inside each setup bootstrap draw."""
    labels = np.asarray(labels, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    reference = np.asarray(reference_predictions, dtype=np.int64)
    groups = np.asarray(setups, dtype=np.int64)
    if not (labels.shape == candidate.shape == reference.shape == groups.shape):
        raise ValueError("Paired accuracy inputs must be aligned vectors")
    unique = np.unique(groups)
    if len(unique) != 12:
        raise ValueError(f"Expected the 12 frozen development setups, got {unique.tolist()}")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    observed_candidate = _metric_values(labels, candidate)
    observed_reference = _metric_values(labels, reference)
    observed = np.subtract(observed_candidate, observed_reference)
    draw_matrix, draw_sha256 = _validated_draws(
        groups, resamples=resamples, seed=seed, draws=draws
    )
    distribution = np.empty((resamples, 3), dtype=np.float64)
    for index, draw in enumerate(draw_matrix):
        selected = _cluster_draw_indices(groups, draw)
        distribution[index] = np.subtract(
            _metric_values(labels[selected], candidate[selected]),
            _metric_values(labels[selected], reference[selected]),
        )
    return {
        "estimate": observed.tolist(),
        "distribution": distribution,
        "endpoint_names": ["all_top1", "newclasses_top1", "macro_f1"],
        "groups": unique.tolist(),
        "samples": len(labels),
        "bootstrap_method": "setup_cluster_resample_with_replacement_centered_basic",
        "draw_matrix_sha256": draw_sha256,
    }


def paired_cost_bootstrap(
    candidate_ms: np.ndarray,
    reference_ms: np.ndarray,
    setups: np.ndarray,
    *,
    resamples: int,
    seed: int,
    draws: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap ratio-of-total measured cost by frozen setup clusters."""
    candidate = np.asarray(candidate_ms, dtype=np.float64)
    reference = np.asarray(reference_ms, dtype=np.float64)
    groups = np.asarray(setups, dtype=np.int64)
    if not (candidate.shape == reference.shape == groups.shape) or candidate.ndim != 1:
        raise ValueError("Paired cost inputs must be aligned vectors")
    if np.any(~np.isfinite(candidate)) or np.any(~np.isfinite(reference)):
        raise ValueError("Cost traces contain non-finite values")
    if np.any(candidate <= 0) or np.any(reference <= 0):
        raise ValueError("Cost traces must be strictly positive")
    unique = np.unique(groups)
    if len(unique) != 12:
        raise ValueError(f"Expected the 12 frozen development setups, got {unique.tolist()}")
    observed = 1.0 - float(candidate.sum() / reference.sum())
    draw_matrix, draw_sha256 = _validated_draws(
        groups, resamples=resamples, seed=seed, draws=draws
    )
    distribution = np.empty(resamples, dtype=np.float64)
    for index, draw in enumerate(draw_matrix):
        selected = _cluster_draw_indices(groups, draw)
        distribution[index] = 1.0 - float(candidate[selected].sum() / reference[selected].sum())
    return {
        "estimate": observed,
        "distribution": distribution,
        "groups": unique.tolist(),
        "bootstrap_method": "setup_cluster_resample_with_replacement_centered_basic",
        "draw_matrix_sha256": draw_sha256,
    }


def _centered_basic_assessment(
    estimate: float,
    distribution: np.ndarray,
    margin: float,
    alpha: float,
) -> tuple[float, float]:
    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("Centered bootstrap assessment requires a finite vector")
    centered = values - float(estimate)
    lower = float(estimate) - float(
        np.quantile(centered, 1.0 - float(alpha), method="higher")
    )
    p_value = float(
        (1.0 + np.sum(centered >= float(estimate) - float(margin)))
        / (len(centered) + 1.0)
    )
    return lower, p_value


def evaluate_frozen_gate(
    accuracy: Mapping[str, Mapping[str, Any]],
    cost: Mapping[str, Mapping[str, Any]],
    *,
    family_alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate the pre-GPU KSP gate; only K32 can support the primary claim."""
    if set(accuracy) != set(VARIANTS) or set(cost) != set(VARIANTS):
        raise ValueError(f"Gate requires exactly {VARIANTS}")
    if family_alpha != 0.05:
        raise ValueError("Frozen AURA-KSP family_alpha must equal 0.05")
    margins = np.asarray([-0.010, -0.015, -0.015, 0.20], dtype=np.float64)
    simultaneous_alpha = family_alpha / len(VARIANTS)
    rows = []
    composite_p = []
    for variant in VARIANTS:
        accuracy_distribution = np.asarray(accuracy[variant]["distribution"], dtype=np.float64)
        cost_distribution = np.asarray(cost[variant]["distribution"], dtype=np.float64)
        if accuracy_distribution.ndim != 2 or accuracy_distribution.shape[1] != 3:
            raise ValueError(f"Invalid accuracy bootstrap for {variant}")
        if len(accuracy_distribution) != 10_000:
            raise ValueError(f"Frozen accuracy bootstrap must contain 10,000 draws: {variant}")
        if cost_distribution.ndim != 1 or len(cost_distribution) != len(accuracy_distribution):
            raise ValueError(f"Invalid cost bootstrap for {variant}")
        combined = np.column_stack([accuracy_distribution, cost_distribution])
        if np.any(~np.isfinite(combined)):
            raise ValueError(f"Non-finite bootstrap values for {variant}")
        estimates = np.asarray(
            [*accuracy[variant]["estimate"], float(cost[variant]["estimate"])],
            dtype=np.float64,
        )
        if estimates.shape != (4,) or np.any(~np.isfinite(estimates)):
            raise ValueError(f"Invalid observed endpoint estimates for {variant}")
        lower = np.empty(4, dtype=np.float64)
        endpoint_p = np.empty(4, dtype=np.float64)
        for endpoint in range(3):
            lower[endpoint], endpoint_p[endpoint] = _centered_basic_assessment(
                estimates[endpoint],
                accuracy_distribution[:, endpoint],
                margins[endpoint],
                simultaneous_alpha,
            )
        regime_bootstraps = cost[variant].get("regime_bootstraps")
        if regime_bootstraps is None:
            lower[3], endpoint_p[3] = _centered_basic_assessment(
                estimates[3], cost_distribution, margins[3], simultaneous_alpha
            )
            cost_regime_assessments = None
        else:
            cost_regime_assessments = {}
            cost_lowers = []
            cost_pvalues = []
            for regime, payload in regime_bootstraps.items():
                regime_estimate = float(payload["estimate"])
                regime_distribution = np.asarray(payload["distribution"], dtype=np.float64)
                if (
                    not np.isfinite(regime_estimate)
                    or regime_distribution.shape != (10_000,)
                ):
                    raise ValueError(f"Invalid cache-regime bootstrap for {variant}/{regime}")
                regime_lower, regime_p = _centered_basic_assessment(
                    regime_estimate,
                    regime_distribution,
                    margins[3],
                    simultaneous_alpha,
                )
                cost_regime_assessments[str(regime)] = {
                    "estimate": regime_estimate,
                    "lcb": regime_lower,
                    "margin_null_p": regime_p,
                }
                cost_lowers.append(regime_lower)
                cost_pvalues.append(regime_p)
            lower[3] = min(cost_lowers)
            endpoint_p[3] = max(cost_pvalues)
        intersection_union_p = float(endpoint_p.max())
        composite_p.append(intersection_union_p)
        endpoint_pass = lower > margins
        endpoint_pass[3] = lower[3] >= margins[3]
        rows.append(
            {
                "variant": variant,
                "estimate": estimates.tolist(),
                "simultaneous_lcb": lower.tolist(),
                "margins": margins.tolist(),
                "endpoint_p": endpoint_p.tolist(),
                "intersection_union_p": intersection_union_p,
                "endpoint_pass": endpoint_pass.tolist(),
                "cost_regime_assessments": cost_regime_assessments,
            }
        )
    adjusted = _holm_adjust(composite_p)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = float(value)
        row["gate_passed"] = bool(all(row["endpoint_pass"]) and value < family_alpha)
        row["scientific_role"] = (
            "primary_prespecified_train_only"
            if row["variant"] == PRIMARY_VARIANT
            else "secondary_curve"
        )
    primary_row = next(row for row in rows if row["variant"] == PRIMARY_VARIANT)
    return {
        "schema": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "method": "AURA-KSP",
        "artifact_version": 2,
        "source_role": "ntu120_xset_train_only_nested_setup_oof",
        "claim_level": "internal_cross_validated_engineering",
        "external_confirmation": False,
        "old_validation_used": False,
        "test_read": False,
        "selection_baseline": "fixed_prior",
        "prior": list(FIXED_PRIOR),
        "fixed_prior": list(FIXED_PRIOR),
        "primary_candidate": PRIMARY_VARIANT,
        "primary_variant": PRIMARY_VARIANT,
        "reference_variant": REFERENCE_VARIANT,
        "family_alpha": family_alpha,
        "simultaneous_alpha": simultaneous_alpha,
        "bootstrap_resamples": 10_000,
        "quantile_method": "higher",
        "bootstrap_method": "centered_setup_cluster_basic_lcb_and_margin_null_tail_p",
        "candidate_results": rows,
        "primary_gate_passed": bool(primary_row["gate_passed"]),
        "selected_candidate": PRIMARY_VARIANT if primary_row["gate_passed"] else None,
        "repeatability_authorized_by_stage1": bool(primary_row["gate_passed"]),
    }


def exact_setup_sign_flip(values: np.ndarray, setups: np.ndarray, margin: float) -> float:
    """Optional one-sided setup-level sign-flip diagnostic for additive endpoints."""
    values = np.asarray(values, dtype=np.float64) + float(margin)
    groups = np.asarray(setups, dtype=np.int64)
    unique = np.unique(groups)
    sums = np.asarray([values[groups == group].sum() for group in unique])
    observed = float(sums.sum())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(unique))))
    null = signs @ sums
    return float((1.0 + np.sum(null >= observed)) / (len(null) + 1.0))
