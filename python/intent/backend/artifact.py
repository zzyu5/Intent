from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from intent.realizer.model import LaunchSpec
from intent.realizer.model import PhysicalPlan


@dataclass(slots=True)
class CompiledArtifact:
    source: str
    intent_ir: str
    plan: PhysicalPlan
    launch: LaunchSpec
    _launcher: Callable[..., object] = field(repr=False)
    backend_ir: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def entry(self) -> Callable[..., None]:
        return self.__call__

    @property
    def ir(self) -> dict[str, str]:
        return {"intent": self.intent_ir, **self.backend_ir}

    def __call__(self, *arguments: Any) -> None:
        compiled_kernel = self._launcher(*arguments)
        asm = getattr(compiled_kernel, "asm", None)
        if not isinstance(asm, dict):
            raise RuntimeError("Triton launch did not return a compiled kernel artifact")
        self.backend_ir = {
            name: value for name, value in asm.items() if isinstance(value, str)
        }
        if not self.backend_ir:
            raise RuntimeError("Triton compiled artifact exposes no textual backend IR")
