from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.cutile import materialize_cutile_artifact


@dataclass(frozen=True, slots=True)
class ResolvedCuTileTarget:
    architecture: str
    device: int

    @property
    def realizer_options(self) -> tuple[str, ...]:
        return (
            f"--architecture={self.architecture}",
            f"--device={self.device}",
        )

    @property
    def realizer_role(self) -> str:
        return "Intent cuTile realizer"

    @property
    def translator_role(self) -> str:
        return "Intent cuTile translator"

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
    ) -> CompiledArtifact:
        return materialize_cutile_artifact(source, module_text, entry_name)


@dataclass(frozen=True, slots=True)
class CuTileTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("cuTile target device must be a non-negative integer")

    def resolve(self) -> ResolvedCuTileTarget:
        import cuda.tile
        import torch

        if not torch.cuda.is_available() or self.device >= torch.cuda.device_count():
            raise RuntimeError("requested cuTile CUDA device is unavailable")
        major, minor = torch.cuda.get_device_capability(self.device)
        return ResolvedCuTileTarget(
            architecture=f"sm_{major}{minor}",
            device=self.device,
        )
