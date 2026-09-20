from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from aura_har.calibration import TemperatureScaler
from aura_har.fusion import REFERENCE_STREAMS, REFERENCE_WEIGHTS, load_prediction_archive

DEVELOPMENT_PRIOR = tuple(float(value / sum(REFERENCE_WEIGHTS)) for value in REFERENCE_WEIGHTS)


def align_prediction_archives(
    paths: Mapping[str, str | Path],
    streams: Sequence[str] = REFERENCE_STREAMS,
) -> dict[str, Any]:
    """Load ordered stream logits and reject any sample or label misalignment."""
    missing = [stream for stream in streams if stream not in paths]
    if missing:
        raise ValueError(f"Missing prediction archives for streams: {missing}")
    archives = {stream: load_prediction_archive(paths[stream]) for stream in streams}
    reference = archives[streams[0]]
    labels = np.asarray(reference["labels"], dtype=np.int64)
    sample_ids = np.asarray(reference["sample_ids"]).astype(str)
    logits_shape = reference["logits"].shape
    if len(logits_shape) != 2:
        raise ValueError(f"Expected N,C logits, got {logits_shape}")
    for stream in streams[1:]:
        archive = archives[stream]
        if archive["logits"].shape != logits_shape:
            raise ValueError(
                f"Logit shape mismatch for {stream}: {archive['logits'].shape} != {logits_shape}"
            )
        if not np.array_equal(np.asarray(archive["labels"], dtype=np.int64), labels):
            raise ValueError(f"Label ordering mismatch for {stream}")
        other_ids = np.asarray(archive["sample_ids"]).astype(str)
        if not np.array_equal(other_ids, sample_ids):
            raise ValueError(f"Sample ordering mismatch for {stream}")
    logits = np.stack(
        [np.asarray(archives[stream]["logits"], dtype=np.float32) for stream in streams],
        axis=1,
    )
    return {
        "logits": logits,
        "labels": labels,
        "sample_ids": sample_ids,
        "streams": tuple(streams),
    }


def normalized_prior(weights: Sequence[float] = REFERENCE_WEIGHTS) -> np.ndarray:
    prior = np.asarray(weights, dtype=np.float64)
    if prior.ndim != 1 or len(prior) == 0:
        raise ValueError("Fusion weights must be a non-empty vector")
    if np.any(prior <= 0) or not np.isfinite(prior).all():
        raise ValueError("Fusion weights must be finite and positive")
    return (prior / prior.sum()).astype(np.float32)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def fit_stream_temperatures(
    logits: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit one scalar per stream. Callers must pass validation data only."""
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C logits, got {logits.shape}")
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    temperatures = []
    for stream_index in range(logits.shape[1]):
        scaler = TemperatureScaler()
        value = scaler.fit(
            torch.as_tensor(logits[:, stream_index], dtype=torch.float32),
            labels_tensor,
            max_iter=max_iter,
        )
        temperatures.append(value)
    return np.asarray(temperatures, dtype=np.float32)


def apply_temperatures(logits: np.ndarray, temperatures: Sequence[float]) -> np.ndarray:
    values = np.asarray(temperatures, dtype=np.float32)
    if logits.ndim != 3 or values.shape != (logits.shape[1],):
        raise ValueError(
            f"Expected {logits.shape[1] if logits.ndim == 3 else '?'} temperatures, "
            f"got {values.shape}"
        )
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError("Temperatures must be finite and positive")
    return np.asarray(logits, dtype=np.float32) / values[None, :, None]


def fuse_logits(logits: np.ndarray, weights: np.ndarray | Sequence[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"Expected N,S,C logits, got {logits.shape}")
    if values.ndim == 1:
        if values.shape != (logits.shape[1],):
            raise ValueError("Static weight count does not match stream count")
        return np.einsum("nsc,s->nc", logits, values, optimize=True)
    if values.shape != logits.shape[:2]:
        raise ValueError(f"Dynamic weights must have shape {logits.shape[:2]}, got {values.shape}")
    return np.einsum("nsc,ns->nc", logits, values, optimize=True)


def build_confusion_graph(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    top_k: int = 3,
    min_weight: float = 0.03,
) -> np.ndarray:
    """Build a symmetric class-confusion graph from development labels only."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have identical shape")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(counts, (labels, predictions), 1)
    totals = np.bincount(labels, minlength=num_classes).astype(np.float64)
    conditional = np.divide(
        counts,
        totals[:, None],
        out=np.zeros_like(counts),
        where=totals[:, None] > 0,
    )
    symmetric = 0.5 * (conditional + conditional.T)
    np.fill_diagonal(symmetric, 0.0)
    keep = np.zeros_like(symmetric, dtype=bool)
    for class_id in range(num_classes):
        order = np.argsort(symmetric[class_id])[::-1][:top_k]
        keep[class_id, order] = symmetric[class_id, order] >= min_weight
    keep = keep | keep.T
    graph = np.where(keep, symmetric, 0.0).astype(np.float32)
    np.fill_diagonal(graph, 0.0)
    return graph


def extract_subject_groups(sample_ids: Sequence[str]) -> np.ndarray:
    """Extract NTU subject IDs (Pxxx) for leakage-safe grouped folds."""
    groups = []
    for sample_id in sample_ids:
        match = re.search(r"P(\d{3})", str(sample_id), flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Cannot extract NTU subject Pxxx from sample_id={sample_id!r}")
        groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def grouped_stratified_folds(
    labels: np.ndarray,
    sample_ids: Sequence[str],
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = extract_subject_groups(sample_ids)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} distinct subjects for grouped cross-fitting")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros(len(labels), dtype=np.uint8)
    folds = []
    for train_indices, heldout_indices in splitter.split(dummy, labels, groups):
        overlap = np.intersect1d(groups[train_indices], groups[heldout_indices])
        if overlap.size:
            raise AssertionError(f"Subject leakage across folds: {overlap.tolist()}")
        folds.append((train_indices, heldout_indices))
    return folds


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=-1)


def _margins(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.sort(np.partition(probabilities, -2, axis=-1)[..., -2:], axis=-1)
    return top_two[..., 1] - top_two[..., 0]


def _pairwise_js(probabilities: np.ndarray) -> np.ndarray:
    values = []
    for first in range(probabilities.shape[1]):
        for second in range(first + 1, probabilities.shape[1]):
            left = np.clip(probabilities[:, first], 1e-12, 1.0)
            right = np.clip(probabilities[:, second], 1e-12, 1.0)
            middle = 0.5 * (left + right)
            divergence = 0.5 * (
                (left * (np.log(left) - np.log(middle))).sum(axis=1)
                + (right * (np.log(right) - np.log(middle))).sum(axis=1)
            )
            values.append(divergence)
    return np.stack(values, axis=1)


def gate_features(
    calibrated_logits: np.ndarray,
    prior: Sequence[float],
    graph: np.ndarray | None,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create label-free gate inputs and optionally apply saved standardization."""
    probabilities = softmax_numpy(calibrated_logits)
    classes = calibrated_logits.shape[-1]
    stream_entropy = _entropy(probabilities) / math.log(classes)
    stream_margin = _margins(probabilities)
    stream_max = probabilities.max(axis=-1)
    pairwise_js = _pairwise_js(probabilities) / math.log(2.0)
    votes = probabilities.argmax(axis=-1)
    vote_counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=classes).max(), 1, votes
    )
    disagreement = 1.0 - vote_counts / probabilities.shape[1]

    prior_logits = fuse_logits(calibrated_logits, np.asarray(prior, dtype=np.float32))
    prior_probabilities = softmax_numpy(prior_logits)
    prior_entropy = _entropy(prior_probabilities) / math.log(classes)
    prior_margin = _margins(prior_probabilities)
    prior_max = prior_probabilities.max(axis=1)
    top_order = np.argsort(prior_probabilities, axis=1)[:, -2:]
    top2 = top_order[:, 0].astype(np.int64)
    top1 = top_order[:, 1].astype(np.int64)
    graph_weight = (
        np.zeros(len(top1), dtype=np.float64)
        if graph is None
        else np.asarray(graph, dtype=np.float64)[top1, top2]
    )
    graph_scale = max(float(np.max(graph)) if graph is not None else 0.0, 1e-6)
    graph_risk = np.clip(graph_weight / graph_scale, 0.0, 1.0)

    numeric = np.concatenate(
        [
            stream_entropy,
            stream_margin,
            stream_max,
            pairwise_js,
            prior_entropy[:, None],
            prior_margin[:, None],
            prior_max[:, None],
            disagreement[:, None],
            graph_weight[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    risk = np.mean(
        np.stack([prior_entropy, 1.0 - prior_margin, disagreement, graph_risk], axis=1),
        axis=1,
    ).astype(np.float32)
    if feature_mean is None:
        feature_mean = numeric.mean(axis=0)
    if feature_std is None:
        feature_std = numeric.std(axis=0)
    feature_std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-6)
    standardized = (numeric - np.asarray(feature_mean, dtype=np.float32)) / feature_std
    return {
        "numeric": standardized.astype(np.float32),
        "top1": top1,
        "top2": top2,
        "risk": risk,
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": feature_std,
        "prior_logits": prior_logits.astype(np.float32),
    }


class AdaptiveFusionGate(nn.Module):
    """Small residual stream gate; pair embeddings are optional for graph conditioning."""

    def __init__(
        self,
        numeric_features: int,
        num_classes: int,
        prior: Sequence[float] = DEVELOPMENT_PRIOR,
        hidden: int = 64,
        dropout: float = 0.1,
        pair_embedding_dim: int = 8,
        residual: bool = True,
        use_pair_embedding: bool = True,
        tau: float = 0.8,
        sharpness: float = 0.1,
    ) -> None:
        super().__init__()
        prior_tensor = torch.as_tensor(normalized_prior(prior), dtype=torch.float32)
        self.register_buffer("prior", prior_tensor)
        self.register_buffer("tau", torch.tensor(float(tau), dtype=torch.float32))
        self.register_buffer("sharpness", torch.tensor(float(sharpness), dtype=torch.float32))
        self.residual = bool(residual)
        self.use_pair_embedding = bool(use_pair_embedding)
        self.num_classes = int(num_classes)
        embedding_features = 0
        if self.use_pair_embedding:
            self.class_embedding = nn.Embedding(num_classes, pair_embedding_dim)
            embedding_features = 2 * pair_embedding_dim
        else:
            self.class_embedding = None
        self.network = nn.Sequential(
            nn.Linear(numeric_features + embedding_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(prior_tensor)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        numeric: torch.Tensor,
        top1: torch.Tensor,
        top2: torch.Tensor,
        risk: torch.Tensor,
    ) -> torch.Tensor:
        features = numeric
        if self.class_embedding is not None:
            features = torch.cat(
                [features, self.class_embedding(top1), self.class_embedding(top2)], dim=1
            )
        delta = self.network(features)
        if not self.residual:
            return torch.softmax(delta, dim=1)
        adaptive = torch.softmax(torch.log(self.prior.clamp_min(1e-8)) + delta, dim=1)
        alpha = torch.sigmoid((risk - self.tau) / self.sharpness.clamp_min(1e-4))[:, None]
        return (1.0 - alpha) * self.prior[None, :] + alpha * adaptive


def train_gate(
    calibrated_logits: np.ndarray,
    labels: np.ndarray,
    inputs: Mapping[str, np.ndarray],
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    device: str | torch.device = "cpu",
) -> tuple[AdaptiveFusionGate, list[dict[str, float]]]:
    """Train a frozen-expert gate on development logits."""
    target_device = torch.device(device)
    seed = int(training_config.get("seed", 42))
    torch.manual_seed(seed)
    model = AdaptiveFusionGate(**model_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    logits_tensor = torch.as_tensor(calibrated_logits, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    numeric = torch.as_tensor(inputs["numeric"], dtype=torch.float32)
    top1 = torch.as_tensor(inputs["top1"], dtype=torch.long)
    top2 = torch.as_tensor(inputs["top2"], dtype=torch.long)
    risk = torch.as_tensor(inputs["risk"], dtype=torch.float32)
    batch_size = int(training_config.get("batch_size", 256))
    epochs = int(training_config.get("epochs", 100))
    lambda_prior = float(training_config.get("lambda_prior", 0.05))
    lambda_dev = float(training_config.get("lambda_dev", 0.01))
    generator = torch.Generator().manual_seed(seed)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(labels_tensor), generator=generator)
        totals = {"loss": 0.0, "ce": 0.0, "prior": 0.0, "dev": 0.0}
        samples = 0
        for start in range(0, len(permutation), batch_size):
            index = permutation[start : start + batch_size]
            batch_logits = logits_tensor[index].to(target_device)
            batch_labels = labels_tensor[index].to(target_device)
            weights = model(
                numeric[index].to(target_device),
                top1[index].to(target_device),
                top2[index].to(target_device),
                risk[index].to(target_device),
            )
            fused = torch.einsum("bsc,bs->bc", batch_logits, weights)
            ce = nn.functional.cross_entropy(fused, batch_labels)
            mean_weights = weights.mean(dim=0).clamp_min(1e-8)
            prior_kl = torch.sum(mean_weights * (torch.log(mean_weights) - torch.log(model.prior)))
            deviation = torch.mean(torch.sum((weights - model.prior[None, :]) ** 2, dim=1))
            loss = ce + lambda_prior * prior_kl + lambda_dev * deviation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(index)
            samples += count
            for name, value in [
                ("loss", loss),
                ("ce", ce),
                ("prior", prior_kl),
                ("dev", deviation),
            ]:
                totals[name] += float(value.detach().cpu()) * count
        history.append(
            {"epoch": float(epoch), **{key: value / samples for key, value in totals.items()}}
        )
    return model.cpu().eval(), history


@torch.inference_mode()
def gate_weights(
    model: AdaptiveFusionGate,
    inputs: Mapping[str, np.ndarray],
    batch_size: int = 2048,
) -> np.ndarray:
    values = []
    for start in range(0, len(inputs["numeric"]), batch_size):
        stop = start + batch_size
        values.append(
            model(
                torch.as_tensor(inputs["numeric"][start:stop], dtype=torch.float32),
                torch.as_tensor(inputs["top1"][start:stop], dtype=torch.long),
                torch.as_tensor(inputs["top2"][start:stop], dtype=torch.long),
                torch.as_tensor(inputs["risk"][start:stop], dtype=torch.float32),
            ).numpy()
        )
    return np.concatenate(values, axis=0).astype(np.float32)


def inverse_entropy_weights(
    calibrated_logits: np.ndarray,
    prior: Sequence[float] = DEVELOPMENT_PRIOR,
    epsilon: float = 1e-3,
) -> np.ndarray:
    probabilities = softmax_numpy(calibrated_logits)
    entropy = _entropy(probabilities) / math.log(calibrated_logits.shape[-1])
    reliability = np.asarray(prior, dtype=np.float64)[None, :] / (entropy + epsilon)
    return (reliability / reliability.sum(axis=1, keepdims=True)).astype(np.float32)


def noisy_or_logits(
    calibrated_logits: np.ndarray,
    prior: Sequence[float] = DEVELOPMENT_PRIOR,
) -> np.ndarray:
    probabilities = np.clip(softmax_numpy(calibrated_logits), 1e-8, 1.0 - 1e-8)
    weights = np.asarray(prior, dtype=np.float64)[None, :, None]
    combined = 1.0 - np.exp(np.sum(weights * np.log1p(-probabilities), axis=1))
    combined /= combined.sum(axis=1, keepdims=True)
    return np.log(np.clip(combined, 1e-12, 1.0)).astype(np.float32)
