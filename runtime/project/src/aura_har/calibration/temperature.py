from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Post-hoc scalar temperature fitted on validation logits only."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(initial_temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu())
