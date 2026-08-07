from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TritonTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("Triton target device must be a non-negative integer")
