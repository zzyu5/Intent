from __future__ import annotations

import ctypes
from dataclasses import dataclass
import statistics

import torch

from .compilation import compile_library
from .contract import MojoFacts
from ..interface import ViewParameter
from ..invocation import ViewFacts, build_invocation_binders, invocation_result
from ..native import NativeABI, NativePreparedRuntime
from ..torch_views import allocate_output, check_abstract_relation, observe_view


_winners: dict[tuple[object, ...], dict[tuple[object, ...], int]] = {}
_candidate_timings: dict[tuple[object, ...], dict[tuple[object, ...], tuple[float, ...]]] = {}

def _timing_samples(measure, arguments: tuple[object, ...], *, samples: int) -> float:
    first = measure(*arguments, 1)
    if first <= 0:
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = min(50, max(1, int(10.0 / first)))
    return statistics.median(measure(*arguments, repetitions) for _ in range(samples))


def _measure_candidates(measurements, arguments: tuple[object, ...]) -> tuple[float, ...]:
    probes = [measure(*arguments, 1) for measure in measurements]
    if any(elapsed <= 0 for elapsed in probes):
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = [min(50, max(1, int(10.0 / elapsed))) for elapsed in probes]
    samples = [[] for _ in measurements]
    for round_index in range(3):
        order = list(range(len(measurements)))
        if round_index == 1:
            order.reverse()
        elif round_index == 2:
            order = order[1:] + order[:1]
        for candidate in order:
            samples[candidate].append(measurements[candidate](*arguments, repetitions[candidate]))
    return tuple(statistics.median(values) for values in samples)


@dataclass
class NativeCall:
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[torch.Tensor, ...]
    key: tuple[object, ...]
    winner: int | None = None

    def choose(self) -> int:
        if self.winner is not None:
            return self.winner
        if self.key not in self.program.winners:
            timings = _measure_candidates(self.program.measurements, self.native_arguments)
            self.program.winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
            self.program.timings[self.key] = timings
        self.winner = self.program.winners[self.key]
        return self.winner

    def launch(self) -> None:
        self.program.functions[self.choose()](*self.native_arguments)

    def benchmark(self) -> float:
        return _timing_samples(self.program.measurements[self.choose()], self.native_arguments, samples=7)

    def result(self):
        return invocation_result(self.outputs)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, abi: NativeABI, facts: MojoFacts, target) -> None:
        self.interface = abi.interface
        self.candidates = facts.candidates
        self.requirements = requirements = abi.requirements
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            check_view_requirements=lambda owner, parameter, facts: requirements.check_view(parameter, facts),
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.offset, facts.dtype, group),
            scalar_key=lambda parameter, value: parameter.dtype.name,
        )
        self._infer = build_invocation_binders(
            self.interface, observe_view=type(self)._abstract_view,
            allocate_output=type(self)._abstract_output,
            check_relation=check_abstract_relation, abstract=True,
        )[0]
        self.compilation = compile_library(facts, target, abi=abi)
        argument_types = abi.argument_types()
        self.functions = []
        self.measurements = []
        for candidate, compilation in zip(self.candidates, self.compilation.libraries):
            function = getattr(compilation.library, candidate.entry)
            function.argtypes = argument_types
            function.restype = None
            self.functions.append(function)
            measure = getattr(compilation.library, candidate.entry + "_benchmark")
            measure.argtypes = [*argument_types, ctypes.c_int64]
            measure.restype = ctypes.c_double
            self.measurements.append(measure)
        self.winners = _winners.setdefault(self.compilation.identity, {})
        self.timings = _candidate_timings.setdefault(self.compilation.identity, {})

    def _view(self, parameter: ViewParameter, tensor) -> ViewFacts:
        facts = observe_view(parameter, tensor, device=torch.device("cpu"))
        if tensor.numel() == 0:
            raise NotImplementedError("Mojo CPU empty-storage pointer ABI is not implemented")
        return facts

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[torch.Tensor, ViewFacts]:
        tensor, facts = allocate_output(parameter, shape, device=torch.device("cpu"))
        if tensor.numel() == 0:
            raise NotImplementedError("Mojo CPU empty-storage pointer ABI is not implemented")
        return tensor, facts

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs, bound.key)

    @staticmethod
    def _abstract_view(owner, parameter: ViewParameter, value) -> ViewFacts:
        return observe_view(parameter, value, device=torch.device("cpu"), abstract=True)

    @staticmethod
    def _abstract_output(owner, parameter: ViewParameter, shape: tuple):
        return allocate_output(parameter, shape, device=torch.device("cpu"), abstract=True)

    def infer_outputs(self, arguments: tuple):
        return self._infer(self, arguments).result()
