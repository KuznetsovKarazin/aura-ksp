from __future__ import annotations

import torch
from torch import nn

from .graph import ntu_adjacency


class TemporalConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTRGC(nn.Module):
    """Channel-wise topology refinement operator from CTR-GCN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        relation_channels = 8 if in_channels <= 16 else max(8, in_channels // 8)
        self.conv_q = nn.Conv2d(in_channels, relation_channels, 1)
        self.conv_k = nn.Conv2d(in_channels, relation_channels, 1)
        self.conv_v = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_relation = nn.Conv2d(relation_channels, out_channels, 1)
        self.tanh = nn.Tanh()

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        query = self.conv_q(x).mean(dim=2).unsqueeze(-1)
        key = self.conv_k(x).mean(dim=2).unsqueeze(-2)
        relation = self.conv_relation(self.tanh(query - key)) * alpha
        relation = relation + adjacency.unsqueeze(0).unsqueeze(0)
        value = self.conv_v(x)
        return torch.einsum("ncuv,nctv->nctu", relation, value)


class UnitGCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.adjacency = nn.Parameter(adjacency.clone())
        self.branches = nn.ModuleList(
            [CTRGC(in_channels, out_channels) for _ in range(adjacency.shape[0])]
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = sum(
            branch(x, self.adjacency[index], self.alpha)
            for index, branch in enumerate(self.branches)
        )
        return self.relu(self.bn(output) + self.residual(x))


class CTRGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn = TemporalConv(out_channels, out_channels, stride=stride)
        if not residual:
            self.residual: nn.Module | None = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + residual)


class CTRGCN(nn.Module):
    """Single-stream CTR-GCN baseline with configurable width.

    Paper-level reproduction requires the matched official preprocessing,
    modality, augmentation, and training schedule documented in the config.
    """

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
        widths = [
            base_channels,
            base_channels,
            base_channels,
            base_channels,
            base_channels * 2,
            base_channels * 2,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
            base_channels * 4,
        ]
        blocks = []
        previous = in_channels
        for index, width in enumerate(widths):
            stride = 2 if index in {4, 7} else 1
            blocks.append(
                CTRGCNBlock(previous, width, adjacency, stride=stride, residual=index != 0)
            )
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(widths[-1], num_classes)

    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        n, c, t, v, m = x.shape
        if (c, v, m) != (self.in_channels, self.num_nodes, self.num_person):
            raise ValueError(
                f"Expected N,{self.in_channels},T,{self.num_nodes},{self.num_person}; got {x.shape}"
            )
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2)
        return x.contiguous().view(n * m, c, t, v), n, m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, n, m = self._normalize_input(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3)).view(n, m, -1).mean(dim=1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
