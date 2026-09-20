from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from aura_har.metrics.classification import softmax_numpy


def selective_classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    coverages: Sequence[float] = (0.8, 0.9, 0.95),
) -> dict[str, Any]:
    probabilities = softmax_numpy(logits)
    predictions = probabilities.argmax(axis=1)
    errors = (predictions != labels).astype(np.float64)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    order = np.argsort(entropy)
    cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    result: dict[str, Any] = {
        "aurc": float(cumulative_risk.mean()),
        "entropy_error_auroc": None,
        "accuracy_at_coverage": {},
    }
    if len(np.unique(errors)) == 2:
        result["entropy_error_auroc"] = float(roc_auc_score(errors, entropy))
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError(f"Coverage must be in (0, 1], got {coverage}")
        count = max(1, int(np.floor(len(errors) * coverage)))
        result["accuracy_at_coverage"][f"{coverage:.2f}"] = float(
            1.0 - errors[order[:count]].mean()
        )
    return result


def paired_stratified_bootstrap(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    num_classes: int,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap stratified by true class without materializing sample indices."""
    labels = np.asarray(labels, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    candidate_predictions = np.asarray(candidate_predictions, dtype=np.int64)
    if not (labels.shape == baseline_predictions.shape == candidate_predictions.shape):
        raise ValueError("Labels and paired predictions must have identical shape")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    predicted_baseline = np.zeros((resamples, num_classes), dtype=np.int32)
    predicted_candidate = np.zeros((resamples, num_classes), dtype=np.int32)
    true_positive_baseline = np.zeros((resamples, num_classes), dtype=np.int32)
    true_positive_candidate = np.zeros((resamples, num_classes), dtype=np.int32)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    for true_class in range(num_classes):
        mask = labels == true_class
        count = int(mask.sum())
        if count == 0:
            continue
        codes = baseline_predictions[mask] * num_classes + candidate_predictions[mask]
        unique, frequencies = np.unique(codes, return_counts=True)
        draws = rng.multinomial(count, frequencies / count, size=resamples)
        for column, code in enumerate(unique):
            baseline_class = int(code // num_classes)
            candidate_class = int(code % num_classes)
            values = draws[:, column]
            predicted_baseline[:, baseline_class] += values
            predicted_candidate[:, candidate_class] += values
            if baseline_class == true_class:
                true_positive_baseline[:, true_class] += values
            if candidate_class == true_class:
                true_positive_candidate[:, true_class] += values

    total = len(labels)
    accuracy_delta = (
        true_positive_candidate.sum(axis=1) - true_positive_baseline.sum(axis=1)
    ) / total
    denominator_baseline = class_counts[None, :] + predicted_baseline
    denominator_candidate = class_counts[None, :] + predicted_candidate
    f1_baseline = np.divide(
        2.0 * true_positive_baseline,
        denominator_baseline,
        out=np.zeros_like(denominator_baseline, dtype=np.float64),
        where=denominator_baseline > 0,
    ).mean(axis=1)
    f1_candidate = np.divide(
        2.0 * true_positive_candidate,
        denominator_candidate,
        out=np.zeros_like(denominator_candidate, dtype=np.float64),
        where=denominator_candidate > 0,
    ).mean(axis=1)
    macro_f1_delta = f1_candidate - f1_baseline

    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    broken = int(np.sum(baseline_correct & ~candidate_correct))
    corrected = int(np.sum(~baseline_correct & candidate_correct))
    discordant = broken + corrected
    mcnemar_p = (
        1.0
        if discordant == 0
        else float(
            binomtest(min(broken, corrected), discordant, 0.5, alternative="two-sided").pvalue
        )
    )

    observed_accuracy_delta = float(candidate_correct.mean() - baseline_correct.mean())
    return {
        "corrected_baseline_errors": corrected,
        "broken_baseline_correct": broken,
        "mcnemar_exact_p": mcnemar_p,
        "bootstrap_resamples": resamples,
        "top1_delta": observed_accuracy_delta,
        "top1_delta_ci95": [
            float(np.quantile(accuracy_delta, 0.025)),
            float(np.quantile(accuracy_delta, 0.975)),
        ],
        "macro_f1_delta_ci95": [
            float(np.quantile(macro_f1_delta, 0.025)),
            float(np.quantile(macro_f1_delta, 0.975)),
        ],
    }


def graph_pair_error_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    graph: np.ndarray,
) -> dict[str, Any]:
    """Evaluate swaps only on class pairs frozen in the validation graph."""
    rows = []
    total_swaps = 0
    total_pair_samples = 0
    for first, second in zip(*np.nonzero(np.triu(graph, 1)), strict=True):
        pair_mask = (labels == first) | (labels == second)
        swaps = int(
            np.sum(
                ((labels == first) & (predictions == second))
                | ((labels == second) & (predictions == first))
            )
        )
        samples = int(pair_mask.sum())
        total_swaps += swaps
        total_pair_samples += samples
        rows.append(
            {
                "class_a": int(first),
                "class_b": int(second),
                "validation_graph_weight": float(graph[first, second]),
                "test_pair_samples": samples,
                "test_swaps": swaps,
                "test_swap_rate": 0.0 if samples == 0 else swaps / samples,
            }
        )
    return {
        "edges": len(rows),
        "swaps": total_swaps,
        "pair_samples_sum": total_pair_samples,
        "micro_swap_rate": (0.0 if total_pair_samples == 0 else total_swaps / total_pair_samples),
        "per_edge": rows,
    }
