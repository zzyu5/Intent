from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.runtime import CompiledArtifact
from intent.runtime.cutile import materialize_cutile_artifact
from intent.runtime.triton import materialize_triton_artifact

from ..specification import GPUCompilationTarget, require_matching_target
from .device import resolve_gpu_device

if TYPE_CHECKING:
    from intent.runtime.contract import ProgramContract


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    materialize: Callable[[str, str, str, int, ProgramContract], CompiledArtifact]


_PROVIDERS = {
    "triton": _Provider("Triton", materialize_triton_artifact),
    "cutile": _Provider("cuTile", materialize_cutile_artifact),
}


def _provider(name: str) -> _Provider:
    if name not in _PROVIDERS:
        raise ValueError(f"unknown GPU provider: {name}")
    return _PROVIDERS[name]


@dataclass(frozen=True, slots=True)
class ResolvedGPUTarget:
    compilation: GPUCompilationTarget
    device: int

    def __post_init__(self) -> None:
        _provider(self.compilation.provider)

    def materialize(self, program) -> CompiledArtifact:
        current = GPUCompilationTarget(self.compilation.provider, resolve_gpu_device(self.device))
        require_matching_target(program.target, current)
        return _provider(self.compilation.provider).materialize(
            program.source, program.ir, program.entry_name, self.device, program._contract)


@dataclass(frozen=True, slots=True)
class GPUTarget(ABC):
    device: int = 0

    @property
    @abstractmethod
    def provider(self) -> str: ...

    def __post_init__(self) -> None:
        provider = _provider(self.provider)
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError(f"{provider.name} target device must be a non-negative integer")

    def resolve(self) -> ResolvedGPUTarget:
        return ResolvedGPUTarget(
            GPUCompilationTarget(self.provider, resolve_gpu_device(self.device)), self.device)
