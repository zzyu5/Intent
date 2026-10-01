from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from intent.runtime import CompiledArtifact
from intent.runtime.cutile import materialize_cutile_artifact
from intent.runtime.tilelang import materialize_tilelang_artifact
from intent.runtime.triton import materialize_triton_artifact

from .device import GpuDeviceCapabilities, resolve_gpu_device


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    required_module: str | None
    materialize: Callable[[str, str, str, int, dict[str, object]], CompiledArtifact]


_PROVIDERS = {
    "triton": _Provider("Triton", None, materialize_triton_artifact),
    "cutile": _Provider("cuTile", "cuda.tile", materialize_cutile_artifact),
    "tilelang": _Provider("TileLang", "tilelang", materialize_tilelang_artifact),
}


def _provider(name: str) -> _Provider:
    if name not in _PROVIDERS:
        raise ValueError(f"unknown GPU provider: {name}")
    return _PROVIDERS[name]


@dataclass(frozen=True, slots=True)
class ResolvedGPUTarget:
    capabilities: GpuDeviceCapabilities
    provider: str

    def __post_init__(self) -> None:
        _provider(self.provider)

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (f"--target={self.provider}", *self.capabilities.compiler_options)

    @property
    def compiler_role(self) -> str:
        return f"Intent {_provider(self.provider).name} compiler"

    def materialize(self, source: str, module_text: str, entry_name: str,
                    metadata: dict[str, object]) -> CompiledArtifact:
        return _provider(self.provider).materialize(
            source, module_text, entry_name, self.capabilities.device, metadata)


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
        provider = _provider(self.provider)
        if provider.required_module is not None:
            import_module(provider.required_module)
        return ResolvedGPUTarget(resolve_gpu_device(self.device), self.provider)
