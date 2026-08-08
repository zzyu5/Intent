from __future__ import annotations

from dataclasses import dataclass

from intent.runtime import CompiledArtifact
from intent.runtime.tilelang import materialize_tilelang_artifact


@dataclass(frozen=True, slots=True)
class ResolvedTileLangTarget:
    architecture: str
    device: int

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=tilelang",
            f"--architecture={self.architecture}",
            f"--device={self.device}",
        )

    @property
    def compiler_role(self) -> str:
        return "Intent TileLang compiler"

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
    ) -> CompiledArtifact:
        return materialize_tilelang_artifact(source, module_text, entry_name)


@dataclass(frozen=True, slots=True)
class TileLangTarget:
    device: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError("TileLang target device must be a non-negative integer")

    def resolve(self) -> ResolvedTileLangTarget:
        import tilelang
        import torch

        if not torch.cuda.is_available() or self.device >= torch.cuda.device_count():
            raise RuntimeError("requested TileLang CUDA device is unavailable")
        major, minor = torch.cuda.get_device_capability(self.device)
        return ResolvedTileLangTarget(
            architecture=f"sm_{major}{minor}",
            device=self.device,
        )
