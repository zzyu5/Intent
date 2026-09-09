from __future__ import annotations

from typing import Protocol, runtime_checkable

from intent.runtime import CompiledArtifact


class ResolvedSourceTarget(Protocol):
    @property
    def compiler_options(self) -> tuple[str, ...]: ...

    @property
    def compiler_role(self) -> str: ...


@runtime_checkable
class ResolvedTarget(ResolvedSourceTarget, Protocol):
    def materialize(
        self,
        source: str,
        module_text: str,
        entry_name: str,
        metadata: dict[str, object],
    ) -> CompiledArtifact: ...


class SourceTarget(Protocol):
    def resolve(self) -> ResolvedSourceTarget: ...


class Target(SourceTarget, Protocol):
    def resolve(self) -> ResolvedTarget: ...
