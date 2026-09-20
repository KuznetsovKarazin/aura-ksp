from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    status: dict[str, Any] = {"seed": seed, "deterministic_requested": deterministic}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        status["torch_seeded"] = True
    except ImportError:
        status["torch_seeded"] = False
    return status


def seed_worker(worker_id: int) -> None:
    """DataLoader worker initializer recommended by PyTorch."""
    del worker_id
    try:
        import torch

        worker_seed = torch.initial_seed() % (2**32)
    except ImportError:
        worker_seed = np.random.SeedSequence().entropy
    np.random.seed(int(worker_seed))
    random.seed(int(worker_seed))


def capture_rng_state() -> dict[str, Any]:
    """Capture host and accelerator RNG state for resumable training."""
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_checkpoint_on_cpu(path: str | Path) -> dict[str, Any]:
    """Deserialize a training checkpoint without relocating host RNG state to CUDA."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG state after loading a checkpoint on any device."""
    if not state:
        return
    import torch

    def cpu_byte_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([cpu_byte_tensor(item) for item in state["cuda"]])


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    """Move tensors nested in optimizer state after CPU-safe checkpoint loading."""
    import torch

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, parameter_state in optimizer.state.items():
        optimizer.state[parameter] = move(parameter_state)
