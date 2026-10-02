from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping
from pathlib import Path

from intent.runtime import CompiledArtifact
from .specification import DSACompilationTarget, require_matching_target


@dataclass(frozen=True, slots=True)
class ResolvedBangCTarget:
    compilation: DSACompilationTarget
    device: int
    neuware: str
    compiler: str | None

    def materialize(self, program) -> CompiledArtifact:
        from intent.runtime.bangc import materialize_bangc_artifact
        require_matching_target(program.target, self.compilation)
        return materialize_bangc_artifact(program.source, program.ir, program._contract, self)


@dataclass(frozen=True, slots=True)
class BangCTarget:
    """BANG C block bindings; source generation also works away from an MLU host."""

    architecture: str = "mtp_372"
    tile: int = 1024
    tile_m: int = 16
    tile_n: int = 64
    tile_k: int = 64
    region_tile: int = 64
    tasks: int = 16
    local_bytes: int = 512 * 1024
    device: int = 0
    neuware: str | Path = "/usr/local/neuware"
    compiler: str | Path | None = None
    shapes: Mapping[str, tuple[int, ...]] | None = None
    strides: Mapping[str, tuple[int, ...]] | None = None

    @classmethod
    def from_program(cls, program, *, device: int, neuware: str | Path,
                     compiler: str | Path | None = None) -> BangCTarget:
        """Select a local runtime while retaining this program's DSA construction bindings."""
        if not isinstance(program.target, DSACompilationTarget):
            raise ValueError("a BANG C runtime requires a DSA generated program")
        fields = asdict(program.target)
        fields["shapes"] = dict(program.target.shapes)
        fields["strides"] = dict(program.target.strides)
        return cls(**fields, device=device, neuware=neuware, compiler=compiler)

    def resolve(self) -> ResolvedBangCTarget:
        if type(self.device) is not int or self.device < 0:
            raise ValueError("BANG C device must be a nonnegative integer")
        shapes = tuple(sorted((name, tuple(shape)) for name, shape in (self.shapes or {}).items()))
        strides = tuple(sorted((name, tuple(values)) for name, values in (self.strides or {}).items()))
        compilation = DSACompilationTarget(
            self.architecture, self.tile, self.tile_m, self.tile_n, self.tile_k,
            self.region_tile, self.tasks, self.local_bytes, shapes, strides)
        return ResolvedBangCTarget(
            compilation, self.device, str(self.neuware),
            str(self.compiler) if self.compiler is not None else None,
        )
