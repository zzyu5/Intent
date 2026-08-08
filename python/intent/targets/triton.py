from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.triton import materialize_triton_artifact


@dataclass(frozen=True, slots=True)
class ResolvedTritonTarget:
    architecture: str
    device: int
    warp_size: int

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=triton",
            f"--architecture={self.architecture}",
            f"--device={self.device}",
            f"--warp-size={self.warp_size}",
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
