from __future__ import annotations

from typing import Any

from torch import nn

from .ctrgcn import CTRGCN
from .ctrgcn_reference import ReferenceCTRGCN
from .stgcn import STGCN


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    data_cfg = config["data"]
    common = {
        "num_classes": int(data_cfg["num_classes"]),
        "num_nodes": int(data_cfg.get("num_joints", 25)),
        "num_person": int(data_cfg.get("num_person", 2)),
        "in_channels": int(model_cfg.get("in_channels", 3)),
        "base_channels": int(model_cfg.get("base_channels", 64)),
        "dropout": float(model_cfg.get("dropout", 0.0)),
    }
    name = str(model_cfg["name"]).lower()
    if name == "stgcn":
        return STGCN(**common)
    if name == "ctrgcn":
        return CTRGCN(**common)
    if name == "ctrgcn_reference":
        return ReferenceCTRGCN(
            **common,
            adaptive=bool(model_cfg.get("adaptive", True)),
        )
    raise ValueError(f"Unknown model: {name}")
