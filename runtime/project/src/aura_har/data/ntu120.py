from __future__ import annotations

from pathlib import Path

from aura_har.data.ntu import NTUSampleName, parse_ntu_name

NTU120_XSUB_TRAIN_SUBJECTS = frozenset(
    {
        1,
        2,
        4,
        5,
        8,
        9,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        25,
        27,
        28,
        31,
        34,
        35,
        38,
        45,
        46,
        47,
        49,
        50,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        70,
        74,
        78,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        89,
        91,
        92,
        93,
        94,
        95,
        97,
        98,
        100,
        103,
    }
)
NTU120_XSET_TRAIN_SETUPS = frozenset(range(2, 33, 2))


def official_split_ntu120(meta: NTUSampleName, protocol: str) -> str:
    """Return the exact official NTU RGB+D 120 benchmark partition."""
    protocol = protocol.lower()
    if not 1 <= meta.action <= 120:
        raise ValueError(f"NTU120 action must be in 1..120: {meta.sample_id}")
    if not 1 <= meta.setup <= 32:
        raise ValueError(f"NTU120 setup must be in 1..32: {meta.sample_id}")
    if protocol == "xsub":
        return "train" if meta.subject in NTU120_XSUB_TRAIN_SUBJECTS else "test"
    if protocol == "xset":
        return "train" if meta.setup in NTU120_XSET_TRAIN_SETUPS else "test"
    raise ValueError(f"Unsupported NTU120 protocol: {protocol}")


def setup_from_sample_id(path_or_name: str | Path) -> int:
    return parse_ntu_name(path_or_name).setup
