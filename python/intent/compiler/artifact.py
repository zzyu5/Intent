from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact
    from intent.targets.base import ResolvedSourceTarget


@dataclass(frozen=True, slots=True)
class GeneratedProgram:
    """Provider source, compiler IR and interface; no native callable or runtime."""

    source: str
    ir: str
    metadata: dict[str, object]
    cache_directory: Path
    entry_name: str
    _target: ResolvedSourceTarget = field(repr=False)

    def materialize(self) -> CompiledArtifact:
        """Bind this generated program with its already resolved target, without recompiling KIR.

        Source-only targets reject materialization. A provider may defer native
        JIT or tuning until invocation; materialization does not launch a kernel.
        """
        from intent.targets.base import ResolvedTarget
        from .toolchain import CompilationStageError

        if not isinstance(self._target, ResolvedTarget):
            raise NotImplementedError("this target provides source generation only")
        try:
            artifact = self._target.materialize(self.source, self.ir, self.entry_name, self.metadata)
        except Exception as error:
            raise CompilationStageError("generated_source_materialization", str(error),
                                        cache_directory=self.cache_directory) from error
        artifact.cache_directory = self.cache_directory
        return artifact
