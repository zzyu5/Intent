from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedBangCTarget:
    architecture: str
    tile: int
    tile_m: int
    tile_n: int
    tile_k: int
    tasks: int
    local_bytes: int
    device: int
    neuware: str
    compiler: str | None

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (
            "--target=bangc", f"--dsa-architecture={self.architecture}",
            f"--dsa-tile={self.tile}", f"--dsa-tile-m={self.tile_m}",
            f"--dsa-tile-n={self.tile_n}", f"--dsa-tile-k={self.tile_k}",
            f"--dsa-tasks={self.tasks}", f"--dsa-local-bytes={self.local_bytes}",
        )

    @property
    def compiler_role(self) -> str:
        return "Intent DSA compiler"

    def materialize(self, source: str, module_text: str, entry_name: str,
                    metadata: dict[str, object]) -> CompiledArtifact:
        from intent.runtime.bangc import materialize_bangc_artifact
        return materialize_bangc_artifact(source, module_text, metadata, self)


@dataclass(frozen=True, slots=True)
class BangCTarget:
    """BANG C block bindings; source generation also works away from an MLU host."""

    architecture: str = "mtp_372"
    tile: int = 1024
    tile_m: int = 16
    tile_n: int = 64
    tile_k: int = 64
    tasks: int = 16
    local_bytes: int = 384 * 1024
    device: int = 0
    neuware: str | Path = "/usr/local/neuware"
    compiler: str | Path | None = None

    def resolve(self) -> ResolvedBangCTarget:
        if self.architecture != "mtp_372":
            raise NotImplementedError("BANG C currently provides an MLU370 implementation profile")
        if any(value <= 0 for value in (self.tile, self.tile_m, self.tile_n, self.tile_k, self.tasks, self.local_bytes)):
            raise ValueError("DSA block and resource bindings must be positive")
        if self.tile % 64 or self.device < 0:
            raise ValueError("DSA vector tile must be divisible by 64 and device must be nonnegative")
        return ResolvedBangCTarget(
            self.architecture, self.tile, self.tile_m, self.tile_n, self.tile_k,
            self.tasks, self.local_bytes, self.device, str(self.neuware),
            str(self.compiler) if self.compiler is not None else None,
        )
