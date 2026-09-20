from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from .graph import ntu_adjacency


def _conv_init(module: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def _bn_init(module: nn.modules.batchnorm._BatchNorm, scale: float) -> None:
    nn.init.constant_(module.weight, scale)
    nn.init.constant_(module.bias, 0)


def _branch_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        _conv_init(module)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0)


class Zero(nn.Module):
    def forward(self, x: torch.Tensor) -> int:
        del x
        return 0


class TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        _conv_init(self.conv)
        _bn_init(self.bn, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    """Official CTR-GCN temporal module.

    With the reference dilations ``(1, 2)``, it has four branches: two
    dilated temporal convolutions, max-pooling, and a strided 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 5,
        stride: int = 1,
        dilations: Sequence[int] = (1, 2),
        residual: bool = False,
    ) -> None:
        super().__init__()
        num_branches = len(dilations) + 2
        if out_channels % num_branches:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by {num_branches} branches"
            )
        branch_channels = out_channels // num_branches
        if isinstance(kernel_size, int):
            kernels = [kernel_size] * len(dilations)
        else:
            kernels = list(kernel_size)
            if len(kernels) != len(dilations):
                raise ValueError("kernel_size and dilations must have equal lengths")

        branches: list[nn.Module] = []
        for kernel, dilation in zip(kernels, dilations, strict=True):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=kernel,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
            )
        branches.extend(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                    nn.BatchNorm2d(branch_channels),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        stride=(stride, 1),
                    ),
                    nn.BatchNorm2d(branch_channels),
                ),
            ]
        )
        self.branches = nn.ModuleList(branches)
        if not residual:
            self.residual: nn.Module = Zero()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, 1, stride=stride)
        self.apply(_branch_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1) + self.residual(x)


class ReferenceCTRGC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        relation_channels = 8 if in_channels in {3, 9} else in_channels // 8
        self.conv1 = nn.Conv2d(in_channels, relation_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, relation_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(relation_channels, out_channels, 1)
        self.tanh = nn.Tanh()
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _conv_init(module)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        x1 = self.conv1(x).mean(dim=2)
        x2 = self.conv2(x).mean(dim=2)
        x3 = self.conv3(x)
        relation = self.conv4(self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2)))
        relation = relation * alpha + adjacency.unsqueeze(0).unsqueeze(0)
        return torch.einsum("ncuv,nctv->nctu", relation, x3)


class ReferenceUnitGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        adaptive: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.num_subset = int(adjacency.shape[0])
        self.convs = nn.ModuleList(
            [ReferenceCTRGC(in_channels, out_channels) for _ in range(self.num_subset)]
        )
        if residual:
            if in_channels == out_channels:
                self.down: nn.Module = nn.Identity()
            else:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
                )
        else:
            self.down = Zero()
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(adjacency.clone())
        else:
            self.register_buffer("A", adjacency.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                _conv_init(module)
            elif isinstance(module, nn.BatchNorm2d):
                _bn_init(module, 1.0)
        _bn_init(self.bn, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.PA if self.adaptive else self.A
        output = sum(conv(x, adjacency[index], self.alpha) for index, conv in enumerate(self.convs))
        return self.relu(self.bn(output) + self.down(x))


class ReferenceCTRGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        self.gcn = ReferenceUnitGCN(in_channels, out_channels, adjacency, adaptive=adaptive)
        self.tcn = MultiScaleTemporalConv(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=stride,
            dilations=(1, 2),
            residual=False,
        )
        if not residual:
            self.residual: nn.Module = Zero()
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, out_channels, 1, stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class ReferenceCTRGCN(nn.Module):
    """Architecturally matched CTR-GCN reference model.

    The default width, graph subsets, block schedule, temporal branches, and
    parameter initialization follow the official ICCV 2021 implementation.
    """

    def __init__(
        self,
        num_classes: int,
        num_nodes: int = 25,
        num_person: int = 2,
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.0,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        adjacency = torch.from_numpy(ntu_adjacency(num_nodes))
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
        blocks: list[nn.Module] = []
        previous = in_channels
        for index, width in enumerate(widths):
            blocks.append(
                ReferenceCTRGCNBlock(
                    previous,
                    width,
                    adjacency,
                    stride=2 if index in {4, 7} else 1,
                    residual=index != 0,
                    adaptive=adaptive,
                )
            )
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.classifier = nn.Linear(widths[-1], num_classes)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_classes))
        _bn_init(self.data_bn, 1.0)

    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        if x.ndim != 5:
            raise ValueError(f"Expected N,C,T,V,M input, got {tuple(x.shape)}")
        n, c, t, v, m = x.shape
        if (c, v, m) != (self.in_channels, self.num_nodes, self.num_person):
            raise ValueError(
                f"Expected N,{self.in_channels},T,{self.num_nodes},{self.num_person}; "
                f"got {tuple(x.shape)}"
            )
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous()
        return x.view(n * m, c, t, v), n, m

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x, n, m = self._normalize_input(x)
        for block in self.blocks:
            x = block(x)
        channels = x.shape[1]
        x = x.view(n, m, channels, -1).mean(dim=3).mean(dim=1)
        return self.dropout(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
