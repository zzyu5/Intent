from __future__ import annotations

from typing import Protocol

from intent.runtime import CompiledArtifact


class ResolvedTarget(Protocol):
    @property
    def realizer_options(self) -> tuple[str, ...]: ...

    @property
    def realizer_role(self) -> str: ...

    @property
    def translator_role(self) -> str: ...

    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
    ) -> CompiledArtifact: ...


class Target(Protocol):
    def resolve(self) -> ResolvedTarget: ...
