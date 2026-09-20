from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from aura_har.adaptive_fusion import fuse_logits, normalized_prior, softmax_numpy
from aura_har.confirmation import protocol_lock_sha256
from aura_har.data.ntu120 import setup_from_sample_id
from aura_har.utils.io import sha256_file

PROTOCOL_ID = "aura-tpc-ntu120-newclasses-xset-v1"
LOCK_SCHEMA = "aura-har.temporal-confirmation-lock.v1"
ARTIFACT_SCHEMA = "aura-har.temporal-progressive-controller.v2"
METHOD_NAME = "AURA-TPC"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")
FIXED_PRIOR = (0.3, 0.3, 0.2, 0.2)
PRIMARY_LABELS = tuple(range(60, 120))
DEVELOPMENT_SETUPS = (4, 6, 8, 12, 14, 16, 18, 22, 24, 26, 28, 32)
OUTER_SETUP_FOLDS = ((4, 12, 18, 26), (6, 14, 22, 28), (8, 16, 24, 32))

FEATURE_NAMES = (
    "joint_entropy",
    "bone_entropy",
    "joint_motion_entropy",
    "bone_motion_entropy",
    "joint_margin",
    "bone_margin",
    "joint_motion_margin",
    "bone_motion_margin",
    "joint_max_probability",
    "bone_max_probability",
    "joint_motion_max_probability",
    "bone_motion_max_probability",
    "pairwise_js_mean",
    "pairwise_js_max",
    "fixed_prior_entropy",
    "fixed_prior_margin",
    "fixed_prior_max_probability",
    "stream_vote_disagreement",
    "budget_fraction",
)


def sample_id_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_setup_groups(sample_ids: Sequence[str]) -> np.ndarray:
    return np.asarray([setup_from_sample_id(str(value)) for value in sample_ids], dtype=np.int64)


def verify_temporal_protocol_lock(
    lock_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    path = Path(lock_path)
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unsupported temporal protocol lock schema")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Temporal protocol ID mismatch")
    if lock.get("status") != "frozen_before_development_and_test":
        raise ValueError("Temporal protocol lock is not frozen")
    root = Path(project_root).resolve()
    mismatches = []
    for relative, expected in lock.get("files", {}).items():
        target = root / relative
        actual = sha256_file(target) if target.is_file() else None
        if actual != expected:
            mismatches.append({"path": relative, "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"Temporal protocol lock mismatch: {mismatches[:3]}")
    return lock


def reject_test_paths(value: Any) -> None:
    """Fail closed if a development input can plausibly address sealed-test material."""
    if isinstance(value, Mapping):
        for nested in value.values():
            reject_test_paths(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            reject_test_paths(nested)
        return
    if not isinstance(value, (str, Path)):
        return
    normalized = Path(str(value)).as_posix().lower()
    forbidden = ("eval_test", "final_test", "final-test", "/test/", "test_manifest")
    if any(token in normalized for token in forbidden):
        raise ValueError(f"Development refuses a sealed-test path: {value}")


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, -2, axis=-1)[..., -2:]
    top_two.sort(axis=-1)
    return top_two[..., 1] - top_two[..., 0]


def temporal_features(
    stream_logits: np.ndarray,
    budget: int,
    max_budget: int = 64,
    prior: Sequence[float] = FIXED_PRIOR,
) -> np.ndarray:
    """Frozen label-free AURA-TPC feature map from the preview logits only."""
    logits = np.asarray(stream_logits, dtype=np.float32)
    if logits.ndim != 3 or logits.shape[1] != len(STREAMS):
        raise ValueError(f"Expected N,{len(STREAMS)},C stream logits, got {logits.shape}")
    if not 0 < int(budget) < int(max_budget):
        raise ValueError("Preview budget must be positive and strictly below max_budget")
    probabilities = softmax_numpy(logits)
    classes = logits.shape[2]
    entropies = _entropy(probabilities) / math.log(classes)
    margins = _margin(probabilities)
    maxima = probabilities.max(axis=2)
    pairwise_js = []
    for first, second in itertools.combinations(range(logits.shape[1]), 2):
        left = np.clip(probabilities[:, first], 1e-12, 1.0)
        right = np.clip(probabilities[:, second], 1e-12, 1.0)
        middle = 0.5 * (left + right)
        value = 0.5 * (
            (left * (np.log(left) - np.log(middle))).sum(axis=1)
            + (right * (np.log(right) - np.log(middle))).sum(axis=1)
        ) / math.log(2.0)
        pairwise_js.append(value)
    js = np.stack(pairwise_js, axis=1)
    fused = fuse_logits(logits, normalized_prior(prior))
    fused_probabilities = softmax_numpy(fused)
    votes = logits.argmax(axis=2)
    vote_support = np.stack(
        [(votes == class_id).sum(axis=1) for class_id in range(classes)], axis=1
    ).max(axis=1)
    features = np.column_stack(
        [
            entropies,
            margins,
            maxima,
            js.mean(axis=1),
            js.max(axis=1),
            _entropy(fused_probabilities) / math.log(classes),
            _margin(fused_probabilities),
            fused_probabilities.max(axis=1),
            1.0 - vote_support / logits.shape[1],
            np.full(len(logits), float(budget) / float(max_budget)),
        ]
    ).astype(np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("Frozen temporal feature schema changed")
    if not np.isfinite(features).all():
        raise ValueError("Temporal features contain non-finite values")
    return features


def fit_ridge_gain(features: np.ndarray, gains: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(gains, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("Ridge features/gains shape mismatch")
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "coefficients": coefficients.astype(np.float32),
    }


def predict_ridge_gain(features: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    standardized = (x - model["feature_mean"]) / model["feature_std"]
    design = np.column_stack([np.ones(len(x)), standardized]).astype(np.float32)
    return np.clip(design @ model["coefficients"], -1.0, 1.0).astype(np.float32)


def group_oof_scores(
    features: np.ndarray, gains: np.ndarray, groups: np.ndarray, alpha: float
) -> np.ndarray:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Group OOF scoring requires at least two groups")
    scores = np.zeros(len(gains), dtype=np.float32)
    covered = np.zeros(len(gains), dtype=np.uint8)
    splitter = LeaveOneGroupOut()
    for train, heldout in splitter.split(features, gains, groups):
        if np.intersect1d(groups[train], groups[heldout]).size:
            raise AssertionError("Group leakage in OOF score fit")
        model = fit_ridge_gain(features[train], gains[train], alpha)
        scores[heldout] = predict_ridge_gain(features[heldout], model)
        covered[heldout] += 1
    if not np.all(covered == 1):
        raise AssertionError("Group OOF scores do not cover every sample exactly once")
    return scores


def select_ridge_alpha(
    features: np.ndarray,
    gains: np.ndarray,
    groups: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[float, np.ndarray, dict[str, float]]:
    if not alpha_grid:
        raise ValueError("alpha_grid must not be empty")
    candidates = []
    cache: dict[float, np.ndarray] = {}
    for value in alpha_grid:
        alpha = float(value)
        scores = group_oof_scores(features, gains, groups, alpha)
        cache[alpha] = scores
        candidates.append((float(np.mean(np.square(scores - gains))), alpha))
    _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected, cache[selected], {str(alpha): value for value, alpha in candidates}


def cluster_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    resamples: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    sample_values = np.asarray(values, dtype=np.float64)
    sample_groups = np.asarray(groups)
    if sample_values.shape != sample_groups.shape or sample_values.ndim != 1:
        raise ValueError("Cluster bootstrap values/groups must be aligned vectors")
    if not 0 < alpha < 1 or resamples <= 0:
        raise ValueError("Invalid cluster bootstrap configuration")
    unique = np.unique(sample_groups)
    if len(unique) < 2:
        raise ValueError("Cluster bootstrap requires at least two groups")
    sums = np.asarray([sample_values[sample_groups == group].sum() for group in unique])
    counts = np.asarray([np.sum(sample_groups == group) for group in unique], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(resamples, len(unique)), endpoint=False)
    distribution = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "mean": float(sample_values.mean()),
        "groups": len(unique),
        "samples": len(sample_values),
        "alpha": float(alpha),
        "interval": [
            float(np.quantile(distribution, alpha)),
            float(np.quantile(distribution, 1.0 - alpha)),
        ],
    }


def cluster_sign_flip_p(
    values: np.ndarray, groups: np.ndarray, *, margin: float = 0.0
) -> float:
    """One-sided exact/randomized setup-level sign-flip p-value for mean > -margin."""
    shifted = np.asarray(values, dtype=np.float64) + float(margin)
    sample_groups = np.asarray(groups)
    unique = np.unique(sample_groups)
    group_sums = np.asarray([shifted[sample_groups == group].sum() for group in unique])
    observed = float(group_sums.sum())
    if len(unique) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(unique))))
    else:
        rng = np.random.default_rng(314159)
        signs = rng.choice((-1.0, 1.0), size=(65_536, len(unique)))
    null_statistics = signs @ group_sums
    return float((1 + np.sum(null_statistics >= observed)) / (len(null_statistics) + 1))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm input must be a vector of probabilities")
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def fit_gain_bins(
    scores: np.ndarray,
    gains: np.ndarray,
    groups: np.ndarray,
    *,
    bins: int,
    minimum_samples: int,
    minimum_groups: int,
    alpha: float,
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if bins < 2:
        raise ValueError("At least two score bins are required")
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    internal = np.unique(np.asarray(quantiles, dtype=np.float64))
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    assignments = np.searchsorted(edges[1:-1], scores, side="right")
    lower = np.full(len(edges) - 1, -np.inf, dtype=np.float64)
    means = np.zeros(len(lower), dtype=np.float64)
    support = np.zeros(len(lower), dtype=np.int64)
    group_support = np.zeros(len(lower), dtype=np.int64)
    for index in range(len(lower)):
        mask = assignments == index
        support[index] = int(mask.sum())
        group_support[index] = len(np.unique(groups[mask]))
        if support[index] < minimum_samples or group_support[index] < minimum_groups:
            continue
        interval = cluster_bootstrap_interval(
            gains[mask],
            groups[mask],
            alpha=alpha,
            resamples=resamples,
            seed=seed + index,
        )
        means[index] = float(interval["mean"])
        lower[index] = float(interval["interval"][0])
    accept = lower > 0.0
    return {
        "edges": edges.astype(np.float32),
        "mean_gain": means.astype(np.float32),
        "lower_gain": lower.astype(np.float32),
        "support": support,
        "group_support": group_support,
        "accept": accept,
    }


def apply_gain_bins(scores: np.ndarray, model: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(model["edges"], dtype=np.float32)
    assignments = np.searchsorted(edges[1:-1], scores, side="right")
    accept = np.asarray(model["accept"], dtype=bool)[assignments]
    lower = np.asarray(model["lower_gain"], dtype=np.float32)[assignments]
    return accept, lower


def _validate_outer_folds(groups: np.ndarray, outer_folds: Sequence[Sequence[int]]) -> None:
    expected = set(map(int, np.unique(groups)))
    flattened = [int(group) for fold in outer_folds for group in fold]
    if set(flattened) != expected or len(flattened) != len(set(flattened)):
        raise ValueError(
            f"Outer folds must partition development setups exactly: {flattened} != {sorted(expected)}"
        )


def nested_cross_fitted_route(
    features: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_folds: Sequence[Sequence[int]],
    alpha_grid: Sequence[float],
    bins: int,
    minimum_bin_samples: int,
    minimum_bin_groups: int,
    switch_alpha: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    _validate_outer_folds(groups, outer_folds)
    candidate_correct = np.asarray(candidate_predictions) == labels
    baseline_correct = np.asarray(baseline_predictions) == labels
    gains = candidate_correct.astype(np.float32) - baseline_correct.astype(np.float32)
    routed = np.asarray(baseline_predictions, dtype=np.int64).copy()
    accepted = np.zeros(len(labels), dtype=bool)
    predicted_gain = np.zeros(len(labels), dtype=np.float32)
    lower_gain = np.full(len(labels), -np.inf, dtype=np.float32)
    covered = np.zeros(len(labels), dtype=np.uint8)
    fold_audit = []
    for fold_index, heldout_groups in enumerate(outer_folds):
        heldout = np.isin(groups, np.asarray(heldout_groups, dtype=np.int64))
        train = ~heldout
        if not heldout.any() or len(np.unique(groups[train])) < 2:
            raise ValueError("Invalid outer fold")
        selected_alpha, inner_scores, inner_mse = select_ridge_alpha(
            features[train], gains[train], groups[train], alpha_grid
        )
        bin_model = fit_gain_bins(
            inner_scores,
            gains[train],
            groups[train],
            bins=bins,
            minimum_samples=minimum_bin_samples,
            minimum_groups=minimum_bin_groups,
            alpha=switch_alpha,
            resamples=bootstrap_resamples,
            seed=seed + 10_000 * fold_index,
        )
        ridge_model = fit_ridge_gain(features[train], gains[train], selected_alpha)
        scores = predict_ridge_gain(features[heldout], ridge_model)
        fold_accept, fold_lower = apply_gain_bins(scores, bin_model)
        heldout_indices = np.flatnonzero(heldout)
        accepted[heldout_indices] = fold_accept
        predicted_gain[heldout_indices] = scores
        lower_gain[heldout_indices] = fold_lower
        routed[heldout_indices[fold_accept]] = candidate_predictions[heldout_indices[fold_accept]]
        covered[heldout_indices] += 1
        fold_audit.append(
            {
                "fold": fold_index + 1,
                "heldout_setups": list(map(int, heldout_groups)),
                "selected_ridge_alpha": selected_alpha,
                "inner_mse": inner_mse,
                "accepted": int(fold_accept.sum()),
                "train_setups": sorted(map(int, np.unique(groups[train]))),
            }
        )
    if not np.all(covered == 1):
        raise AssertionError("Nested outer predictions must cover every sample exactly once")
    if accepted.any() and not np.all(lower_gain[accepted] > 0.0):
        raise AssertionError("AURA-TPC accepted a switch without a positive gain lower bound")
    return {
        "predictions": routed,
        "accepted": accepted,
        "predicted_gain": predicted_gain,
        "lower_gain": lower_gain,
        "fold_audit": fold_audit,
        "paired_gain": (routed == labels).astype(np.float32)
        - baseline_correct.astype(np.float32),
    }


def fit_final_router(
    features: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    alpha_grid: Sequence[float],
    bins: int,
    minimum_bin_samples: int,
    minimum_bin_groups: int,
    switch_alpha: float,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    gains = (candidate_predictions == labels).astype(np.float32) - (
        baseline_predictions == labels
    ).astype(np.float32)
    selected_alpha, oof_scores, _ = select_ridge_alpha(features, gains, groups, alpha_grid)
    bin_model = fit_gain_bins(
        oof_scores,
        gains,
        groups,
        bins=bins,
        minimum_samples=minimum_bin_samples,
        minimum_groups=minimum_bin_groups,
        alpha=switch_alpha,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    ridge_model = fit_ridge_gain(features, gains, selected_alpha)
    return {
        "ridge_alpha": np.asarray(selected_alpha, dtype=np.float32),
        **ridge_model,
        **{f"bin_{key}": value for key, value in bin_model.items()},
    }


def apply_final_router(features: np.ndarray, model: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ridge_model = {
        "feature_mean": model["feature_mean"],
        "feature_std": model["feature_std"],
        "coefficients": model["coefficients"],
    }
    scores = predict_ridge_gain(features, ridge_model)
    bin_model = {
        key: model[f"bin_{key}"]
        for key in ("edges", "mean_gain", "lower_gain", "support", "group_support", "accept")
    }
    accepted, lower = apply_gain_bins(scores, bin_model)
    if accepted.any() and not np.all(lower[accepted] > 0.0):
        raise AssertionError("Stored router violates the positive lower-bound rule")
    return {"accepted": accepted, "predicted_gain": scores, "lower_gain": lower}


def verify_cost_audit(
    audit: Mapping[str, Any], protocol_id: str = PROTOCOL_ID
) -> dict[str, float]:
    if audit.get("schema") != "aura-har.temporal-cost-audit.v1":
        raise ValueError("Unsupported cost-audit schema")
    if audit.get("protocol_id") != protocol_id or audit.get("verified") is not True:
        raise ValueError("Cost audit is not verified for this protocol")
    if audit.get("boundary") != "end_to_end_batch1_same_device":
        raise ValueError("Gate requires same-device batch-1 end-to-end costs")
    baseline = float(audit.get("baseline_median_ms", 0.0))
    if baseline <= 0 or not math.isfinite(baseline):
        raise ValueError("Invalid baseline cost")
    candidates = audit.get("candidate_median_ms")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Cost audit has no candidate costs")
    ratios = {}
    for identifier, value in candidates.items():
        cost = float(value)
        if cost <= 0 or not math.isfinite(cost):
            raise ValueError(f"Invalid candidate cost: {identifier}")
        ratios[str(identifier)] = cost / baseline
    return ratios


def assess_candidate(
    routed: Mapping[str, Any],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    preview_cost_ratio: float,
    alpha: float,
    bootstrap_resamples: int,
    seed: int,
    primary_noninferiority_margin: float,
) -> dict[str, Any]:
    paired_gain = np.asarray(routed["paired_gain"], dtype=np.float64)
    accepted = np.asarray(routed["accepted"], dtype=bool)
    primary = np.isin(labels, np.asarray(PRIMARY_LABELS))
    if not primary.any():
        raise ValueError("Development predictions contain no A61-A120 samples")
    cost_saving = 1.0 - (float(preview_cost_ratio) + (~accepted).astype(np.float64))
    primary_interval = cluster_bootstrap_interval(
        paired_gain[primary],
        groups[primary],
        alpha=alpha,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    overall_interval = cluster_bootstrap_interval(
        paired_gain,
        groups,
        alpha=alpha,
        resamples=bootstrap_resamples,
        seed=seed + 1,
    )
    cost_interval = cluster_bootstrap_interval(
        cost_saving,
        groups,
        alpha=alpha,
        resamples=bootstrap_resamples,
        seed=seed + 2,
    )
    return {
        "primary_top1_gain": primary_interval,
        "overall_top1_gain": overall_interval,
        "end_to_end_cost_saving": cost_interval,
        "noninferiority_p": cluster_sign_flip_p(
            paired_gain[primary],
            groups[primary],
            margin=primary_noninferiority_margin,
        ),
        "accepted_samples": int(accepted.sum()),
        "acceptance_fraction": float(accepted.mean()),
        "minimum_switch_lower_gain": (
            None if not accepted.any() else float(np.min(routed["lower_gain"][accepted]))
        ),
    }


def gate_candidates(
    rows: list[dict[str, Any]],
    *,
    family_alpha: float,
    primary_noninferiority_margin: float,
    overall_noninferiority_margin: float,
    minimum_cost_saving: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No temporal candidates were assessed")
    adjusted = holm_adjust([float(row["assessment"]["noninferiority_p"]) for row in rows])
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        assessment = row["assessment"]
        primary_lower = float(assessment["primary_top1_gain"]["interval"][0])
        overall_lower = float(assessment["overall_top1_gain"]["interval"][0])
        cost_lower = float(assessment["end_to_end_cost_saving"]["interval"][0])
        switches_valid = (
            assessment["accepted_samples"] > 0
            and float(assessment["minimum_switch_lower_gain"]) > 0.0
        )
        row["holm_adjusted_noninferiority_p"] = float(adjusted_p)
        row["gate_passed"] = bool(
            adjusted_p < family_alpha
            and primary_lower > -primary_noninferiority_margin
            and overall_lower > -overall_noninferiority_margin
            and cost_lower >= minimum_cost_saving
            and switches_valid
        )
    eligible = [row for row in rows if row["gate_passed"]]
    if not eligible:
        return {
            "passed": False,
            "selected_candidate": None,
            "baseline": "fixed_prior",
            "rule": "fail_closed_no_candidate_passed_all_coprimary_guards",
        }
    selected = max(
        eligible,
        key=lambda row: (
            row["assessment"]["end_to_end_cost_saving"]["interval"][0],
            row["assessment"]["primary_top1_gain"]["interval"][0],
            -int(row["budget"]),
            str(row["policy"]),
        ),
    )
    return {
        "passed": True,
        "selected_candidate": str(selected["id"]),
        "baseline": "fixed_prior",
        "rule": "max_cost_lcb_then_primary_gain_lcb_among_multiplicity_safe_candidates",
    }


def validate_test_artifact(
    artifact: Mapping[str, np.ndarray], lock_path: str | Path, config: Mapping[str, Any]
) -> None:
    def scalar(name: str) -> Any:
        if name not in artifact:
            raise ValueError(f"Temporal artifact misses {name}")
        return np.asarray(artifact[name]).item()

    if str(scalar("artifact_schema")) != ARTIFACT_SCHEMA or int(scalar("artifact_version")) != 2:
        raise ValueError("Refusing unsupported or legacy AURA-TPC artifact")
    if str(scalar("method")) != METHOD_NAME:
        raise ValueError("Refusing an artifact from another method")
    if str(scalar("selection_baseline")) != "fixed_prior":
        raise ValueError("Refusing an artifact not selected against fixed prior")
    if not bool(scalar("gate_passed")):
        raise ValueError("Development gate failed; sealed test remains unread")
    if str(scalar("source_role")) != "train_only_nested_oof":
        raise ValueError("Artifact provenance is not nested train-only OOF")
    if str(scalar("protocol_id")) != PROTOCOL_ID:
        raise ValueError("Artifact protocol ID mismatch")
    if str(scalar("protocol_lock_sha256")) != protocol_lock_sha256(lock_path):
        raise ValueError("Artifact was frozen under another protocol lock")
    expected_prior = normalized_prior(config["fusion"]["prior"])
    if not np.allclose(np.asarray(artifact["prior"]), expected_prior, atol=0.0, rtol=0.0):
        raise ValueError("Artifact comparator prior differs from fixed prior")
    if tuple(map(int, np.asarray(artifact["primary_labels"]))) != PRIMARY_LABELS:
        raise ValueError("Artifact primary endpoint differs from A61-A120")
