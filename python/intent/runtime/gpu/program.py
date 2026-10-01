from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

from ..artifact import CompiledArtifact
from ..source import load_python_source
from .interface import BoundInvocation, GPUInterface


@dataclass(slots=True)
class LaunchResult:
    replay: Callable[[], object] | None
    kernel: object = None


class Provider(Protocol):
    def launch(self, invocation: BoundInvocation) -> LaunchResult: ...

    def tuning_configurations(self, invocation: BoundInvocation) -> tuple: ...


class PreparedCall:
    """A call bound to these tensors and scalars, including invocation-owned workspace."""

    def __init__(self, program: GPUProgram, invocation: BoundInvocation) -> None:
        self.program = program
        self.invocation = invocation
        self._launch: LaunchResult | None = None

    @property
    def outputs(self) -> tuple:
        return self.invocation.outputs

    def result(self):
        return self.invocation.result()

    def launch(self):
        import torch

        with self.program.invocation_context():
            if torch.cuda.current_device() != self.program.interface.device:
                with torch.cuda.device(self.program.interface.device):
                    return self._invoke()
            return self._invoke()

    def _invoke(self):
        if self._launch is None or self._launch.replay is None:
            self._launch = self.program.provider.launch(self.invocation)
        else:
            self._launch.replay()
        if self.program.artifact is not None:
            self.program.artifact._capture_backend_ir(self._launch.kernel)
        return self._launch.kernel

    __call__ = launch


class GPUProgram:
    def __init__(self, interface: GPUInterface, provider: Provider) -> None:
        self.interface = interface
        self.provider = provider
        self.artifact: CompiledArtifact | None = None
        self.invocation_context = nullcontext

    def prepare(self, *arguments, outputs: tuple | None = None,
                explicit_outputs: bool = False) -> PreparedCall:
        with self.invocation_context():
            invocation = self.interface.bind(arguments, outputs=outputs, explicit_outputs=explicit_outputs)
        return PreparedCall(self, invocation)

    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None) -> PreparedCall:
        return self.prepare(*arguments, outputs=outputs)

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()

    def launch(self, *arguments):
        return self.prepare(*arguments, explicit_outputs=True).launch()

    def tuning_configurations(self, *arguments):
        invocation = self.interface.bind(arguments, explicit_outputs=True)
        return self.provider.tuning_configurations(invocation)


def materialize_gpu_program(*, provider_name: str, provider_type, source: str,
                            module_text: str, metadata: dict, entry_name: str,
                            device: int, backend_ir_collector=None) -> CompiledArtifact:
    if metadata["provider"] != provider_name:
        raise ValueError(f"{provider_name} runtime cannot load {metadata['provider']} metadata")
    interface = GPUInterface(metadata, device)
    namespace = load_python_source(target_name=provider_name, source=source, entry_name=entry_name)
    provider = provider_type(interface, namespace, metadata[provider_name])
    program = GPUProgram(interface, provider)
    artifact = CompiledArtifact(source=source, mlir=module_text, device=device,
                                runtime=program,
                                _backend_ir_collector=backend_ir_collector)
    program.artifact = artifact
    return artifact
