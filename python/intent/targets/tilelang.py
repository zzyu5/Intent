from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.tilelang import materialize_tilelang_artifact
from intent.targets.gpu import GpuDeviceCapabilities, resolve_gpu_device


@dataclass(frozen=True, slots=True)
class ResolvedTileLangTarget:
    capabilities: GpuDeviceCapabilities

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=tilelang",
            *self.capabilities.compiler_options,
        )

    @property
    def compiler_role(self) -> str:
        return "Intent TileLang compiler"

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
        metadata: dict[str, object],
    ) -> CompiledArtifact:
        return materialize_tilelang_artifact(
            source, module_text, entry_name, self.capabilities.device
        )


@dataclass(frozen=True, slots=True)
class TileLangTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("TileLang target device must be a non-negative integer")

    def resolve(self) -> ResolvedTileLangTarget:
        import tilelang
        return ResolvedTileLangTarget(resolve_gpu_device(self.device))
