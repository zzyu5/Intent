from __future__ import annotations

from typing import Protocol

from intent.runtime import CompiledArtifact


class ResolvedTarget(Protocol):
    @property
    def compiler_options(self) -> tuple[str, ...]: ...

    @property
    def compiler_role(self) -> str: ...

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
        metadata: dict[str, object],
    ) -> CompiledArtifact: ...


class Target(Protocol):
    def resolve(self) -> ResolvedTarget: ...
