from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.triton import materialize_triton_artifact
from intent.targets.gpu import GpuDeviceCapabilities, resolve_gpu_device


_PARAMETER_ROLES = frozenset(
    {
        "ownership_m",
        "ownership_n",
        "reduction",
        "scan_chunk",
        "traversal_workers",
        "traversal_group",
        "resident_workers",
    }
)


@dataclass(frozen=True, slots=True)
class TritonParameterBinding:
    role: str
    value: int
    dimension: int | None = None

    def __post_init__(self) -> None:
        if self.role not in _PARAMETER_ROLES:
            raise ValueError(f"unknown Triton physical parameter role: {self.role!r}")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, int)
            or self.value <= 0
        ):
            raise ValueError("Triton physical parameter values must be positive integers")
        if self.dimension is not None and (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension <= 0
        ):
            raise ValueError("Triton physical parameter dimensions must be positive integers")

    def encode(self) -> str:
        dimension = "" if self.dimension is None else f"@{self.dimension}"
        return f"{self.role}{dimension}={self.value}"


@dataclass(frozen=True, slots=True)
class TritonConfig:
    parameters: tuple[TritonParameterBinding, ...]
    num_warps: int
    num_stages: int
    num_ctas: int = 1

    def __post_init__(self) -> None:
        identities = tuple(
            (binding.role, binding.dimension) for binding in self.parameters
        )
        if len(set(identities)) != len(identities):
            raise ValueError("one Triton config cannot bind a typed parameter role twice")
        for name, value in (
            ("num_warps", self.num_warps),
            ("num_stages", self.num_stages),
            ("num_ctas", self.num_ctas),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def encode(self) -> str:
        fields = [binding.encode() for binding in self.parameters]
        fields.extend(
            (
                f"num_warps={self.num_warps}",
                f"num_stages={self.num_stages}",
                f"num_ctas={self.num_ctas}",
            )
        )
        return ",".join(fields)


@dataclass(frozen=True, slots=True)
class ResolvedTritonTarget:
    capabilities: GpuDeviceCapabilities
    configs: tuple[TritonConfig, ...]

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=triton",
            *self.capabilities.compiler_options,
            *(f"--triton-config={config.encode()}" for config in self.configs),
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
        return materialize_triton_artifact(source, module_text, entry_name)


@dataclass(frozen=True, slots=True)
class TritonTarget:
    device: int = 0
    configs: tuple[TritonConfig, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("Triton target device must be a non-negative integer")
        if not isinstance(self.configs, tuple) or not all(
            isinstance(config, TritonConfig) for config in self.configs
        ):
            raise TypeError("Triton target configs must be a tuple of TritonConfig values")

    def resolve(self) -> ResolvedTritonTarget:
        return ResolvedTritonTarget(resolve_gpu_device(self.device), self.configs)
