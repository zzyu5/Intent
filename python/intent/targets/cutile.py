from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.cutile import materialize_cutile_artifact
from intent.targets.gpu import GpuDeviceCapabilities, resolve_gpu_device


@dataclass(frozen=True, slots=True)
class ResolvedCuTileTarget:
    capabilities: GpuDeviceCapabilities

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=cutile",
            *self.capabilities.compiler_options,
        )

    @property
    def compiler_role(self) -> str:
        return "Intent cuTile compiler"

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
        metadata: dict[str, object],
    ) -> CompiledArtifact:
        return materialize_cutile_artifact(
            source, module_text, entry_name, self.capabilities.device
        )


@dataclass(frozen=True, slots=True)
class CuTileTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("cuTile target device must be a non-negative integer")

    def resolve(self) -> ResolvedCuTileTarget:
        import cuda.tile
        return ResolvedCuTileTarget(resolve_gpu_device(self.device))
