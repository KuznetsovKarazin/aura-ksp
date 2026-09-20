from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from aura_har.adaptive_fusion import fuse_logits, normalized_prior, softmax_numpy


def ambiguity_score(
    calibrated_logits: np.ndarray,
    prior: Sequence[float],
) -> np.ndarray:
    """Compute a label-free ambiguity score in [0, 1]."""
    logits = np.asarray(calibrated_logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C logits, got {logits.shape}")
    if logits.shape[1] < 2 or logits.shape[2] < 2:
        raise ValueError("Ambiguity scoring requires at least two streams and classes")
    prior_array = normalized_prior(prior)
    if prior_array.shape != (logits.shape[1],):
        raise ValueError("Prior length does not match the stream count")

    fused_probabilities = softmax_numpy(fuse_logits(logits, prior_array))
    clipped = np.clip(fused_probabilities, 1e-12, 1.0)
    normalized_entropy = -(clipped * np.log(clipped)).sum(axis=1) / np.log(logits.shape[2])
    top_two = np.sort(
        np.partition(fused_probabilities, -2, axis=1)[:, -2:],
        axis=1,
    )
    margin = top_two[:, 1] - top_two[:, 0]

    votes = logits.argmax(axis=2)
    maximum_vote_count = np.max(
        np.stack([(votes == class_id).sum(axis=1) for class_id in range(logits.shape[2])]),
        axis=0,
    )
    disagreement = 1.0 - maximum_vote_count / logits.shape[1]
    score = (normalized_entropy + (1.0 - margin) + disagreement) / 3.0
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def fit_prediction_precision(
    calibrated_logits: np.ndarray,
    labels: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Estimate P(y=c | stream predicts c) with accuracy-centred shrinkage."""
    logits = np.asarray(calibrated_logits, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C logits, got {logits.shape}")
    if labels_array.shape != (len(logits),):
        raise ValueError("Labels must have shape N")
    if not np.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("shrinkage must be finite and positive")
    if np.any((labels_array < 0) | (labels_array >= logits.shape[2])):
        raise ValueError("Labels contain a class outside the logit range")

    predictions = logits.argmax(axis=2)
    table = np.empty((logits.shape[1], logits.shape[2]), dtype=np.float32)
    for stream_index in range(logits.shape[1]):
        stream_predictions = predictions[:, stream_index]
        global_accuracy = float(np.mean(stream_predictions == labels_array))
        for class_id in range(logits.shape[2]):
            predicted_class = stream_predictions == class_id
            count = int(predicted_class.sum())
            correct = int(np.sum(labels_array[predicted_class] == class_id))
            table[stream_index, class_id] = (correct + shrinkage * global_accuracy) / (
                count + shrinkage
            )
    return table


def ambiguity_reliability_weights(
    calibrated_logits: np.ndarray,
    prior: Sequence[float],
    precision_table: np.ndarray,
    gamma: float,
    ambiguity_threshold: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend the prior with prediction-conditioned reliability on ambiguous samples."""
    logits = np.asarray(calibrated_logits, dtype=np.float32)
    prior_array = normalized_prior(prior)
    table = np.asarray(precision_table, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C logits, got {logits.shape}")
    if prior_array.shape != (logits.shape[1],):
        raise ValueError("Prior length does not match the stream count")
    if table.shape != logits.shape[1:]:
        raise ValueError(f"Precision table must have shape {logits.shape[1:]}, got {table.shape}")
    if not np.isfinite(table).all() or np.any((table < 0) | (table > 1)):
        raise ValueError("Precision table must contain finite probabilities")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if not np.isfinite(ambiguity_threshold) or not 0 <= ambiguity_threshold <= 1:
        raise ValueError("ambiguity_threshold must be in [0, 1]")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")

    predictions = logits.argmax(axis=2)
    reliability = np.stack(
        [table[index, predictions[:, index]] for index in range(logits.shape[1])],
        axis=1,
    )
    target = prior_array[None, :] * np.power(np.clip(reliability, 1e-6, 1.0), gamma)
    target /= target.sum(axis=1, keepdims=True)

    score = ambiguity_score(logits, prior_array)
    adapted = score >= ambiguity_threshold
    alpha = strength * adapted.astype(np.float32)
    weights = (1.0 - alpha[:, None]) * prior_array[None, :] + alpha[:, None] * target
    return weights.astype(np.float32), score, adapted
