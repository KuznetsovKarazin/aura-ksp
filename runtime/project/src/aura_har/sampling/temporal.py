from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np


class TemporalSampler(Protocol):
    def __call__(
        self, sequence: np.ndarray, length: int, budget: int, rng: np.random.Generator
    ) -> np.ndarray: ...


def _linspace_indices(length: int, budget: int) -> np.ndarray:
    if length <= 0 or budget <= 0:
        raise ValueError(f"length and budget must be positive, got {length}, {budget}")
    return np.rint(np.linspace(0, length - 1, budget)).astype(np.int64)


@dataclass
class UniformSampler:
    random_offset: bool = False

    def __call__(
        self, sequence: np.ndarray, length: int, budget: int, rng: np.random.Generator
    ) -> np.ndarray:
        del sequence
        if length < budget or not self.random_offset:
            return _linspace_indices(length, budget)
        boundaries = np.linspace(0, length, budget + 1, dtype=np.int64)
        indices = []
        for start, stop in pairwise(boundaries):
            stop = max(stop, start + 1)
            indices.append(int(rng.integers(start, stop)))
        return np.asarray(indices, dtype=np.int64)


@dataclass
class RandomSampler:
    def __call__(
        self, sequence: np.ndarray, length: int, budget: int, rng: np.random.Generator
    ) -> np.ndarray:
        del sequence
        if length < budget:
            return np.sort(rng.integers(0, length, size=budget, endpoint=False)).astype(np.int64)
        return np.sort(rng.choice(length, size=budget, replace=False)).astype(np.int64)


@dataclass
class FullPadSampler:
    """Keep all frames up to budget and repeat the last valid frame when shorter."""

    def __call__(
        self, sequence: np.ndarray, length: int, budget: int, rng: np.random.Generator
    ) -> np.ndarray:
        del sequence, rng
        if length >= budget:
            return np.arange(budget, dtype=np.int64)
        return np.concatenate(
            [np.arange(length, dtype=np.int64), np.full(budget - length, length - 1, np.int64)]
        )


@dataclass
class MotionSampler:
    mode: str = "segment"
    min_distance: int = 1

    @staticmethod
    def score(sequence: np.ndarray, length: int) -> np.ndarray:
        clip = sequence[:, :length]
        velocity = np.diff(clip, axis=1, prepend=clip[:, :1])
        return np.square(velocity).sum(axis=(0, 2, 3))

    def __call__(
        self, sequence: np.ndarray, length: int, budget: int, rng: np.random.Generator
    ) -> np.ndarray:
        del rng
        if length <= budget:
            return _linspace_indices(length, budget)
        scores = self.score(sequence, length)
        if self.mode == "segment":
            selected = []
            for segment in np.array_split(np.arange(length), budget):
                selected.append(int(segment[np.argmax(scores[segment])]))
            return np.asarray(sorted(selected), dtype=np.int64)
        if self.mode == "topk":
            return np.sort(np.argsort(scores)[-budget:]).astype(np.int64)
        if self.mode == "distance":
            selected: list[int] = []
            for candidate in np.argsort(scores)[::-1]:
                if all(abs(int(candidate) - other) >= self.min_distance for other in selected):
                    selected.append(int(candidate))
                if len(selected) == budget:
                    break
            if len(selected) < budget:
                for candidate in _linspace_indices(length, budget):
                    if int(candidate) not in selected:
                        selected.append(int(candidate))
                    if len(selected) == budget:
                        break
            return np.asarray(sorted(selected), dtype=np.int64)
        raise ValueError(f"Unknown motion mode: {self.mode}")


def build_sampler(name: str, train: bool = False, **kwargs: object) -> TemporalSampler:
    name = name.lower()
    if name == "uniform":
        return UniformSampler(random_offset=bool(kwargs.get("random_offset", train)))
    if name == "random":
        return RandomSampler()
    if name in {"full", "full_pad"}:
        return FullPadSampler()
    if name in {"motion", "motion_segment"}:
        return MotionSampler(mode="segment")
    if name == "motion_topk":
        return MotionSampler(mode="topk")
    if name == "motion_distance":
        return MotionSampler(mode="distance", min_distance=int(kwargs.get("min_distance", 2)))
    raise ValueError(f"Unknown sampler: {name}")
