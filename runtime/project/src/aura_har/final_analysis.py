from __future__ import annotations

import re
from itertools import pairwise
from typing import Any

import numpy as np

from aura_har.metrics import softmax_numpy


def extract_subject_ids(sample_ids: np.ndarray) -> np.ndarray:
    subjects = []
    for sample_id in sample_ids:
        match = re.search(r"P(\d{3})", str(sample_id))
        if match is None:
            raise ValueError(f"Cannot extract subject from sample ID: {sample_id}")
        subjects.append(int(match.group(1)))
    return np.asarray(subjects, dtype=np.int64)


def reliability_rows(
    logits: np.ndarray,
    labels: np.ndarray,
    method: str,
    bins: int = 15,
) -> list[dict[str, Any]]:
    probabilities = softmax_numpy(logits)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    rows = []
    for index, (lower, upper) in enumerate(pairwise(np.linspace(0.0, 1.0, bins + 1)), start=1):
        mask = (confidence > lower) & (confidence <= upper)
        rows.append(
            {
                "method": method,
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_confidence": None if not mask.any() else float(confidence[mask].mean()),
                "accuracy": None if not mask.any() else float(correct[mask].mean()),
            }
        )
    return rows


def ambiguity_decile_rows(
    ambiguity: np.ndarray,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    edges = np.quantile(ambiguity, np.linspace(0.0, 1.0, 11))
    rows = []
    for index, (lower, upper) in enumerate(pairwise(edges), start=1):
        mask = (ambiguity >= lower) & (ambiguity <= upper if index == 10 else ambiguity < upper)
        baseline_accuracy = float(np.mean(baseline_predictions[mask] == labels[mask]))
        candidate_accuracy = float(np.mean(candidate_predictions[mask] == labels[mask]))
        rows.append(
            {
                "decile": index,
                "lower": float(lower),
                "upper": float(upper),
                "samples": int(mask.sum()),
                "baseline_top1": baseline_accuracy,
                "candidate_top1": candidate_accuracy,
                "top1_delta": candidate_accuracy - baseline_accuracy,
                "prediction_change_fraction": float(
                    np.mean(candidate_predictions[mask] != baseline_predictions[mask])
                ),
            }
        )
    return rows


def retrospective_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    minimum_top1_gain: float,
    minimum_macro_f1_gain: float,
    maximum_nll_increase: float,
) -> dict[str, Any]:
    top1_delta = float(candidate["top1"] - baseline["top1"])
    macro_f1_delta = float(candidate["macro_f1"] - baseline["macro_f1"])
    nll_delta = float(candidate["nll"] - baseline["nll"])
    checks = {
        "top1": top1_delta >= minimum_top1_gain,
        "macro_f1": macro_f1_delta >= minimum_macro_f1_gain,
        "nll": nll_delta <= maximum_nll_increase,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "top1_delta": top1_delta,
        "macro_f1_delta": macro_f1_delta,
        "nll_delta": nll_delta,
    }
