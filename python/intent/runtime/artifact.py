from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any


BackendIRCollector = Callable[[object], dict[str, str]]


@dataclass(slots=True)
class CompiledArtifact:
    source: str
    mlir: str
    _launcher: Callable[..., object] = field(repr=False)
    _runner: Callable[..., object] = field(repr=False)
    _backend_ir_collector: BackendIRCollector | None = field(repr=False)
    backend_ir: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def entry(self) -> Callable[..., None]:
        return self.__call__

    @property
    def ir(self) -> dict[str, str]:
        return {"intent": self.mlir, **self.backend_ir}

    def run(self, *arguments: Any) -> object:
        return self._runner(*arguments)

    def __call__(self, *arguments: Any) -> None:
        compiled_kernel = self._launcher(*arguments)
        if self._backend_ir_collector is not None:
            self.backend_ir = self._backend_ir_collector(compiled_kernel)
