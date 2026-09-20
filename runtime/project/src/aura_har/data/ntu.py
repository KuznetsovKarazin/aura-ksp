from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

NTU_NAME = re.compile(
    r"^S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})$"
)

NTU60_XSUB_TRAIN_SUBJECTS = frozenset(
    {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
)


@dataclass(frozen=True)
class NTUSampleName:
    sample_id: str
    setup: int
    camera: int
    subject: int
    replication: int
    action: int

    @property
    def label(self) -> int:
        return self.action - 1

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def parse_ntu_name(path_or_name: str | Path) -> NTUSampleName:
    stem = Path(path_or_name).stem
    match = NTU_NAME.match(stem)
    if not match:
        raise ValueError(f"Invalid NTU sample name: {path_or_name}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    return NTUSampleName(sample_id=stem, **values)


def official_split(meta: NTUSampleName, dataset: str, protocol: str) -> str:
    dataset = dataset.lower()
    protocol = protocol.lower()
    if dataset == "ntu60" and protocol == "xsub":
        return "train" if meta.subject in NTU60_XSUB_TRAIN_SUBJECTS else "test"
    if dataset == "ntu60" and protocol == "xview":
        return "train" if meta.camera in {2, 3} else "test"
    if dataset == "ntu120":
        raise NotImplementedError(
            "NTU RGB+D 120 splits are intentionally not inferred. "
            "Add and test the exact official X-Sub120/X-Set protocol first."
        )
    raise ValueError(f"Unsupported dataset/protocol: {dataset}/{protocol}")


def _body_score(sequence: np.ndarray) -> float:
    """Prioritize long, moving tracks when more than two body IDs are present."""
    valid = np.any(np.abs(sequence) > 0, axis=(1, 2))
    valid_count = float(valid.sum())
    if sequence.shape[0] < 2:
        return valid_count
    velocity = np.diff(sequence, axis=0)
    motion = float(np.square(velocity).sum())
    return valid_count * 1_000_000.0 + motion


def read_skeleton_file(
    path: str | Path, num_joints: int = 25, max_persons: int = 2
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Parse an NTU raw skeleton file into C,T,V,M float32 format.

    Body IDs are tracked across frames. When more than ``max_persons`` tracks are
    observed, the longest/moving tracks are retained deterministically.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError(f"Empty skeleton file: {source}")
        num_frames = int(first.strip())
        tracks: dict[str, np.ndarray] = {}
        frame_has_body = np.zeros(num_frames, dtype=bool)

        for frame_index in range(num_frames):
            line = handle.readline()
            if not line:
                raise ValueError(f"Unexpected EOF before frame {frame_index}: {source}")
            body_count = int(line.strip())
            frame_has_body[frame_index] = body_count > 0
            for body_number in range(body_count):
                body_info = handle.readline().split()
                if not body_info:
                    raise ValueError(f"Missing body metadata at frame {frame_index}: {source}")
                body_id = body_info[0] or f"anonymous-{body_number}"
                joint_count_line = handle.readline()
                if not joint_count_line:
                    raise ValueError(f"Missing joint count at frame {frame_index}: {source}")
                joint_count = int(joint_count_line.strip())
                if body_id not in tracks:
                    tracks[body_id] = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
                for joint_index in range(joint_count):
                    values = handle.readline().split()
                    if len(values) < 3:
                        raise ValueError(
                            f"Malformed joint {joint_index} at frame {frame_index}: {source}"
                        )
                    if joint_index < num_joints:
                        tracks[body_id][frame_index, joint_index] = np.asarray(
                            values[:3], dtype=np.float32
                        )

    ranked = sorted(tracks.items(), key=lambda item: (-_body_score(item[1]), item[0]))
    output = np.zeros((3, num_frames, num_joints, max_persons), dtype=np.float32)
    for person_index, (_, track) in enumerate(ranked[:max_persons]):
        output[:, :, :, person_index] = track.transpose(2, 0, 1)
    metadata = {
        "num_frames": num_frames,
        "observed_body_ids": len(tracks),
        "retained_persons": min(len(tracks), max_persons),
        "empty_frames": int((~frame_has_body).sum()),
    }
    return output, frame_has_body, metadata


def load_ignore_list(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    ignored: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if candidate:
                ignored.add(Path(candidate).stem)
    return ignored


def validate_disjoint(manifests: Iterable[Iterable[dict]]) -> None:
    seen: set[str] = set()
    for rows in manifests:
        current = {str(row["sample_id"]) for row in rows}
        overlap = seen.intersection(current)
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Manifest overlap detected: {preview}")
        seen.update(current)
