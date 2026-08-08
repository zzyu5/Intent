from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResolvedTritonTarget:
    architecture: str
    device: int
    warp_size: int


@dataclass(frozen=True, slots=True)
class TritonTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("Triton target device must be a non-negative integer")

    def resolve(self) -> ResolvedTritonTarget:
        import torch

        if not torch.cuda.is_available() or self.device >= torch.cuda.device_count():
            raise RuntimeError("requested Triton CUDA device is unavailable")
        major, minor = torch.cuda.get_device_capability(self.device)
        return ResolvedTritonTarget(
            architecture=f"sm_{major}{minor}",
            device=self.device,
            warp_size=32,
        )
