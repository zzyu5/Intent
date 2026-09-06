from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from typing import Any


BackendIRCollector = Callable[[object], dict[str, str]]


class ParameterRole(IntEnum):
    OWNERSHIP_M = 0
    OWNERSHIP_N = 1
    REDUCTION = 2
    SCAN_CHUNK = 3
    PROVIDER_WARPS = 4
    PROVIDER_STAGES = 5
    PROVIDER_CTAS = 6
    PROVIDER_THREADS = 7
    TRAVERSAL_WORKERS = 8
    TRAVERSAL_GROUP = 9
    RESIDENT_WORKERS = 10
    FULL_COVERAGE = 11
    REDUCTION_OUTER = 12
    REDUCTION_INNER = 13
    PROVIDER_ACCESS_FORM = 14
    PROVIDER_OCCUPANCY = 15


@dataclass(frozen=True, slots=True)
class TuningParameter:
    name: str
    role: ParameterRole
    category: int
    candidates: tuple[int, ...]
    dimension: int | None
    source: tuple[int, int, bool] | None
    # Tensor argument position in artifact.entry, followed by its logical axis.
    view_axis: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class TuningConfiguration:
    parameters: tuple[TuningParameter, ...]
    values: tuple[int, ...]


@dataclass(slots=True)
class CompiledArtifact:
    source: str
    mlir: str
    device: int
    _launcher: Callable[..., object] = field(repr=False)
    _runner: Callable[..., object] = field(repr=False)
    _backend_ir_collector: BackendIRCollector | None = field(repr=False)
    _namespace: dict[str, object] = field(repr=False)
    backend_ir: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def entry(self) -> Callable[..., None]:
        return self.__call__

    @property
    def ir(self) -> dict[str, str]:
        return {"intent": self.mlir, **self.backend_ir}

    def run(self, *arguments: Any) -> object:
        return self._invoke(self._runner, arguments)

    def tuning_configurations(
        self, *arguments: Any,
    ) -> tuple[TuningConfiguration, ...]:
        function = self._namespace.get("tuning_configurations")
        if function is None:
            raise NotImplementedError(
                "this provider does not export structured tuning configurations"
            )
        return self._invoke(function, arguments)

    def __call__(self, *arguments: Any) -> None:
        compiled_kernel = self._invoke(self._launcher, arguments)
        if self._backend_ir_collector is not None:
            self.backend_ir = self._backend_ir_collector(compiled_kernel)

    def _invoke(
        self,
        function: Callable[..., object],
        arguments: tuple[Any, ...],
    ) -> object:
        import torch

        expected = torch.device("cuda", self.device)
        for index, argument in enumerate(arguments):
            if isinstance(argument, torch.Tensor) and argument.device != expected:
                raise ValueError(
                    f"compiled artifact is bound to {expected}, but tensor argument "
                    f"{index} is on {argument.device}"
                )
        with torch.cuda.device(expected):
            return function(*arguments)
