from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aura_har.adaptive_fusion import fuse_logits, normalized_prior, softmax_numpy

FEATURE_NAMES = (
    "baseline_probability",
    "alternative_probability",
    "probability_gap",
    "normalized_entropy",
    "stream_disagreement",
    "baseline_vote_fraction",
    "alternative_vote_fraction",
    "baseline_weighted_vote",
    "alternative_weighted_vote",
    "baseline_stream_probability_mean",
    "alternative_stream_probability_mean",
    "baseline_stream_probability_max",
    "alternative_stream_probability_max",
    "stream_probability_difference",
)


@dataclass(frozen=True)
class CandidateRows:
    sample_index: np.ndarray
    baseline_class: np.ndarray
    alternative_class: np.ndarray
    features: np.ndarray


def _take_class(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values)), classes]


def build_candidate_rows(
    stream_logits: np.ndarray,
    prior: Sequence[float],
    sample_mask: np.ndarray | None = None,
) -> tuple[CandidateRows, np.ndarray]:
    """Create label-free directed class-pair alternatives for each sample.

    Alternatives are the fixed-prior runner-up and every distinct stream vote. The fixed
    prediction itself is never a candidate, so abstention exactly preserves the baseline.
    """
    logits = np.asarray(stream_logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C stream logits, got {logits.shape}")
    weights = normalized_prior(prior)
    if weights.shape != (logits.shape[1],):
        raise ValueError("Prior length does not match stream count")
    indices = np.arange(len(logits))
    if sample_mask is not None:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != (len(logits),):
            raise ValueError("sample_mask must have shape N")
        indices = indices[mask]
    selected = logits[indices]
    fixed_logits = fuse_logits(selected, weights)
    fixed_probabilities = softmax_numpy(fixed_logits)
    stream_probabilities = np.stack(
        [softmax_numpy(selected[:, stream]) for stream in range(selected.shape[1])], axis=1
    )
    baseline = fixed_logits.argmax(axis=1)
    runner_up = np.argsort(fixed_logits, axis=1)[:, -2]
    votes = selected.argmax(axis=2)
    clipped = np.clip(fixed_probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / np.log(logits.shape[2])
    maximum_votes = np.max(
        np.stack([(votes == c).sum(axis=1) for c in range(logits.shape[2])], axis=1), axis=1
    )
    disagreement = 1.0 - maximum_votes / logits.shape[1]

    row_sample: list[int] = []
    row_baseline: list[int] = []
    row_alternative: list[int] = []
    row_features: list[list[float]] = []
    for local_index, global_index in enumerate(indices):
        alternatives = sorted({int(runner_up[local_index]), *map(int, votes[local_index])})
        base = int(baseline[local_index])
        for alternative in alternatives:
            if alternative == base:
                continue
            base_votes = votes[local_index] == base
            alternative_votes = votes[local_index] == alternative
            base_stream_probability = stream_probabilities[local_index, :, base]
            alternative_stream_probability = stream_probabilities[local_index, :, alternative]
            base_probability = float(fixed_probabilities[local_index, base])
            alternative_probability = float(fixed_probabilities[local_index, alternative])
            row_sample.append(int(global_index))
            row_baseline.append(base)
            row_alternative.append(alternative)
            row_features.append(
                [
                    base_probability,
                    alternative_probability,
                    base_probability - alternative_probability,
                    float(entropy[local_index]),
                    float(disagreement[local_index]),
                    float(base_votes.mean()),
                    float(alternative_votes.mean()),
                    float(weights[base_votes].sum()),
                    float(weights[alternative_votes].sum()),
                    float(base_stream_probability.mean()),
                    float(alternative_stream_probability.mean()),
                    float(base_stream_probability.max()),
                    float(alternative_stream_probability.max()),
                    float((alternative_stream_probability - base_stream_probability).mean()),
                ]
            )
    rows = CandidateRows(
        sample_index=np.asarray(row_sample, dtype=np.int64),
        baseline_class=np.asarray(row_baseline, dtype=np.int64),
        alternative_class=np.asarray(row_alternative, dtype=np.int64),
        features=np.asarray(row_features, dtype=np.float32),
    )
    return rows, fuse_logits(logits, weights)


def paired_gain_targets(rows: CandidateRows, labels: np.ndarray) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=np.int64)
    truth = labels_array[rows.sample_index]
    baseline_correct = rows.baseline_class == truth
    alternative_correct = rows.alternative_class == truth
    return alternative_correct.astype(np.float32) - baseline_correct.astype(np.float32)


def fit_benefit_model(
    rows: CandidateRows,
    gains: np.ndarray,
    num_classes: int,
    ridge: float,
    pair_shrinkage: float,
) -> dict[str, np.ndarray]:
    if len(rows.features) == 0:
        raise ValueError("No class-pair candidates available")
    if ridge <= 0 or pair_shrinkage <= 0:
        raise ValueError("ridge and pair_shrinkage must be positive")
    y = np.asarray(gains, dtype=np.float64)
    feature_mean = rows.features.mean(axis=0).astype(np.float32)
    feature_std = np.maximum(rows.features.std(axis=0), 1e-6).astype(np.float32)
    standardized = (rows.features - feature_mean) / feature_std
    design = np.column_stack([np.ones(len(standardized)), standardized]).astype(np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    global_prediction = design @ coefficients
    residual = y - global_prediction
    global_variance = float(max(np.mean(np.square(residual)), 1e-8))

    pair_code = rows.baseline_class * num_classes + rows.alternative_class
    pair_count = num_classes * num_classes
    support = np.bincount(pair_code, minlength=pair_count).astype(np.int64)
    residual_sum = np.bincount(pair_code, weights=residual, minlength=pair_count)
    offset = residual_sum / (support + float(pair_shrinkage))
    centred = residual - offset[pair_code]
    residual_squares = np.bincount(pair_code, weights=np.square(centred), minlength=pair_count)
    variance = (residual_squares + pair_shrinkage * global_variance) / (support + pair_shrinkage)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coefficients": coefficients.astype(np.float32),
        "pair_offset": offset.astype(np.float32),
        "pair_support": support,
        "pair_variance": variance.astype(np.float32),
        "global_residual_variance": np.asarray(global_variance, dtype=np.float32),
    }


def predict_benefit(
    rows: CandidateRows,
    model: dict[str, np.ndarray],
    num_classes: int,
    pair_shrinkage: float,
    z_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = (rows.features - model["feature_mean"]) / model["feature_std"]
    design = np.column_stack([np.ones(len(standardized)), standardized]).astype(np.float32)
    pair_code = rows.baseline_class * num_classes + rows.alternative_class
    expected = design @ model["coefficients"] + model["pair_offset"][pair_code]
    effective_support = model["pair_support"][pair_code] + float(pair_shrinkage)
    standard_error = np.sqrt(model["pair_variance"][pair_code] / effective_support)
    lower_bound = expected - float(z_value) * standard_error
    return np.clip(expected, -1.0, 1.0), np.clip(lower_bound, -1.0, 1.0)


def route_with_abstention(
    fixed_logits: np.ndarray,
    rows: CandidateRows,
    expected_gain: np.ndarray,
    lower_bound: np.ndarray,
    pair_support: np.ndarray,
    num_classes: int,
    minimum_pair_support: int,
    gain_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Switch at most one directed pair per sample; otherwise retain fixed logits exactly."""
    routed = np.asarray(fixed_logits, dtype=np.float32).copy()
    chosen_alternative = np.full(len(routed), -1, dtype=np.int64)
    chosen_expected = np.zeros(len(routed), dtype=np.float32)
    chosen_lower = np.zeros(len(routed), dtype=np.float32)
    pair_code = rows.baseline_class * num_classes + rows.alternative_class
    eligible = (pair_support[pair_code] >= int(minimum_pair_support)) & (
        lower_bound > float(gain_margin)
    )
    for row_index in np.flatnonzero(eligible):
        sample = int(rows.sample_index[row_index])
        if chosen_alternative[sample] >= 0 and lower_bound[row_index] <= chosen_lower[sample]:
            continue
        chosen_alternative[sample] = int(rows.alternative_class[row_index])
        chosen_expected[sample] = float(expected_gain[row_index])
        chosen_lower[sample] = float(lower_bound[row_index])
    switched = chosen_alternative >= 0
    for sample in np.flatnonzero(switched):
        baseline = int(np.argmax(fixed_logits[sample]))
        alternative = int(chosen_alternative[sample])
        routed[sample, baseline], routed[sample, alternative] = (
            routed[sample, alternative],
            routed[sample, baseline],
        )
    return routed, switched, chosen_alternative, chosen_expected


def fit_and_route(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    evaluation_logits: np.ndarray,
    prior: Sequence[float],
    parameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    num_classes = int(train_logits.shape[2])
    train_rows, _ = build_candidate_rows(train_logits, prior)
    model = fit_benefit_model(
        train_rows,
        paired_gain_targets(train_rows, train_labels),
        num_classes,
        float(parameters["ridge"]),
        float(parameters["pair_shrinkage"]),
    )
    evaluation_rows, fixed_logits = build_candidate_rows(evaluation_logits, prior)
    expected, lower = predict_benefit(
        evaluation_rows,
        model,
        num_classes,
        float(parameters["pair_shrinkage"]),
        float(parameters["z_value"]),
    )
    routed, switched, alternative, chosen_expected = route_with_abstention(
        fixed_logits,
        evaluation_rows,
        expected,
        lower,
        model["pair_support"],
        num_classes,
        int(parameters["minimum_pair_support"]),
        float(parameters["gain_margin"]),
    )
    decisions = {
        "switched": switched,
        "alternative_class": alternative,
        "expected_gain": chosen_expected,
    }
    return routed, model, decisions
