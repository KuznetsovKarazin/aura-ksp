from __future__ import annotations

import torch
from torch import nn

from .graph import ntu_adjacency


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.num_subsets = int(adjacency.shape[0])
        self.projection = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        n, _, t, v = projected.shape
        projected = projected.view(n, self.num_subsets, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", projected, self.adjacency)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.temporal = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual: nn.Module | None = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return self.relu(self.temporal(self.gcn_bn(self.gcn(x))) + residual)


class STGCN(nn.Module):
    """Compact, transparent ST-GCN used for tests and control experiments."""

    def __init__(
        self,
        num_classes: int,
        num_nodes: int = 25,
        num_person: int = 2,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        adjacency = torch.tensor(ntu_adjacency(num_nodes))
        self.num_nodes = num_nodes
        self.num_person = num_person
        self.in_channels = in_channels
        self.data_bn = nn.BatchNorm1d(num_person * num_nodes * in_channels)
        channels = [base_channels, base_channels, base_channels * 2, base_channels * 2]
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, channels[0], adjacency, residual=False),
                STGCNBlock(channels[0], channels[1], adjacency),
                STGCNBlock(channels[1], channels[2], adjacency, stride=2),
                STGCNBlock(channels[2], channels[3], adjacency),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        n, c, t, v, m = x.shape
        if (c, v, m) != (self.in_channels, self.num_nodes, self.num_person):
            raise ValueError(
                f"Expected N,{self.in_channels},T,{self.num_nodes},{self.num_person}; got {x.shape}"
            )
        normalized = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        normalized = self.data_bn(normalized)
        normalized = normalized.view(n, m, v, c, t).permute(0, 1, 3, 4, 2)
        return normalized.contiguous().view(n * m, c, t, v), n, m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, n, m = self._normalize_input(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3)).view(n, m, -1).mean(dim=1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
