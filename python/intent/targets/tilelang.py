from dataclasses import dataclass
from typing import ClassVar

from .gpu import GPUTarget


@dataclass(frozen=True, slots=True)
class TileLangTarget(GPUTarget):
    provider: ClassVar[str] = "tilelang"
