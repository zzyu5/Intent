from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.triton import materialize_triton_artifact
from intent.targets.gpu import GpuDeviceCapabilities, resolve_gpu_device


@dataclass(frozen=True, slots=True)
class ResolvedTritonTarget:
    capabilities: GpuDeviceCapabilities

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=triton",
            *self.capabilities.compiler_options,
        )

    @property
    def compiler_role(self) -> str:
        return "Intent Triton compiler"

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
    ) -> CompiledArtifact:
        return materialize_triton_artifact(
            source, module_text, entry_name, self.capabilities.device
        )


@dataclass(frozen=True, slots=True)
class TritonTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("Triton target device must be a non-negative integer")
    def resolve(self) -> ResolvedTritonTarget:
        return ResolvedTritonTarget(resolve_gpu_device(self.device))
