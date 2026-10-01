from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


BackendIRCollector = Callable[[object], dict[str, str]]


class ArtifactRuntime(Protocol):
    def run(self, *arguments: Any) -> object: ...

    def launch(self, *arguments: Any) -> object: ...


@runtime_checkable
class PreparedCall(Protocol):
    """Invocation-owned arguments and outputs, using the provider's launch semantics.

    result() returns output containers; it does not perform synchronization.
    """

    def launch(self) -> object:
        """Execute with the bound arguments and the provider's synchronization semantics."""
        ...

    def result(self) -> object:
        """Return the runtime's output containers without launching or synchronizing."""
        ...


@runtime_checkable
class PreparedRuntime(Protocol):
    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None) -> PreparedCall: ...


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
    PROVIDER_LOAD_POLICY = 16


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
    runtime: ArtifactRuntime = field(repr=False, metadata={
        "doc": "Explicit provider runtime implementing run/launch. Backend extensions use this object; "
               "additional capabilities depend on the provider, not a generated module namespace."
    })
    _backend_ir_collector: BackendIRCollector | None = field(repr=False)
    device_type: str = field(default="cuda", kw_only=True)
    cache_directory: Path | None = field(default=None, kw_only=True)
    backend_ir: dict[str, str] = field(default_factory=dict, init=False)
    _backend_ir_kernel: object = field(default_factory=object, init=False, repr=False)

    @property
    def entry(self) -> Callable[..., None]:
        return self.__call__

    @property
    def ir(self) -> dict[str, str]:
        return {"intent": self.mlir, **self.backend_ir}

    def run(self, *arguments: Any) -> object:
        """Allocate declared Out buffers and execute with inputs in declaration order."""
        return self._invoke(self.runtime.run, arguments)

    @property
    def interface(self):
        """Read this runtime's typed invocation interface; unsupported runtimes raise NotImplementedError."""
        if not hasattr(self.runtime, "interface"):
            raise NotImplementedError("this runtime does not expose a typed invocation interface")
        return self.runtime.interface

    def prepare(self, *arguments: Any, outputs: tuple | None = None) -> PreparedCall:
        """Bind arguments and allocate invocation-owned outputs/workspace without executing.

        The returned call's launch performs any first-use JIT/tuning and execution.
        GPU calls use the provider's current stream; CPU calls complete their join;
        BANG C calls synchronize their queue. result() only returns output containers.
        Explicit outputs replace declared Out buffers, in declaration order.
        Prepare again when arguments or their shape/stride metadata change, and keep
        their allocations alive while using the prepared call.
        """
        if not isinstance(self.runtime, PreparedRuntime):
            raise NotImplementedError("this runtime does not expose prepared calls")
        return self._invoke(lambda *args: self.runtime.prepare_call(args, outputs=outputs), arguments)

    def as_torch_op(self, name: str):
        """Return a PyTorch CustomOpDef for this allocating GPU call.

        Only read-only In tensors, scalar inputs and fresh Out tensors are supported;
        InOut and output aliases are unsupported. The fake implementation uses the
        declared interface without provider execution or data-pointer access.
        Backward is not inferred; authors may use the result's register_autograd.
        """
        from .torch import register_operator

        return register_operator(self, name)

    def tuning_configurations(
        self, *arguments: Any,
    ) -> tuple[TuningConfiguration, ...]:
        if not hasattr(self.runtime, "tuning_configurations"):
            raise NotImplementedError(
                "this provider does not export structured tuning configurations"
            )
        return self._invoke(self.runtime.tuning_configurations, arguments)

    def __call__(self, *arguments: Any) -> None:
        compiled_kernel = self._invoke(self.runtime.launch, arguments)
        self._capture_backend_ir(compiled_kernel)

    def _capture_backend_ir(self, compiled_kernel: object) -> None:
        if self._backend_ir_collector is not None and (
            compiled_kernel is not self._backend_ir_kernel or not self.backend_ir
        ):
            self.backend_ir = self._backend_ir_collector(compiled_kernel)
            self._backend_ir_kernel = compiled_kernel

    def _invoke(
        self,
        function: Callable[..., object],
        arguments: tuple[Any, ...],
    ) -> object:
        if self.device_type in {"cpu", "mlu"}:
            return function(*arguments)
        import torch

        expected = torch.device(self.device_type, self.device) if self.device_type == "cuda" else torch.device(self.device_type)
        for index, argument in enumerate(arguments):
            if isinstance(argument, torch.Tensor) and argument.device != expected:
                raise ValueError(
                    f"compiled artifact is bound to {expected}, but tensor argument "
                    f"{index} is on {argument.device}"
                )
        if expected.type == "cuda":
            if torch.cuda.current_device() == self.device:
                return function(*arguments)
            with torch.cuda.device(self.device):
                return function(*arguments)
        raise NotImplementedError(f"artifact invocation for {expected.type}")
