from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(edges):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(error)


def classification_metrics(
    labels: np.ndarray, logits: np.ndarray, num_classes: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    probabilities = softmax_numpy(logits)
    predictions = probabilities.argmax(axis=1)
    class_ids = np.arange(num_classes)
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    per_class = f1_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    true_prob = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    metrics: dict[str, Any] = {
        "samples": len(labels),
        "top1": float((predictions == labels).mean()),
        "macro_f1": float(per_class.mean()),
        "nll": float(-np.log(true_prob).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": expected_calibration_error(probabilities, labels),
        "per_class_f1": [float(value) for value in per_class],
    }
    return metrics, matrix, predictions
