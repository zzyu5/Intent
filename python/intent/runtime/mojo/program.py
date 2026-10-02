from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, replace
import os
import statistics
from threading import RLock

import torch

from .compilation import compile_portfolio, runtime_inputs_current
from .contract import MojoFacts
from ..interface import ViewParameter
from ..invocation import ViewFacts, build_invocation_binders, invocation_result
from ..native import NativeABI, NativePreparedRuntime
from ..native_artifact import load_native_library
from ..torch_views import allocate_output, check_abstract_relation, observe_view
from ..diagnostics import CacheObservation, CandidateObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


_winners: dict[tuple[object, ...], dict[tuple[object, ...], int]] = {}
_candidate_timings: dict[tuple[object, ...], dict[tuple[object, ...], tuple[float, ...]]] = {}

def _timing_samples(measure, arguments: tuple[object, ...], *, samples: int) -> float:
    first = measure(*arguments, 1)
    if first <= 0:
        raise RuntimeError("native monotonic timing returned a non-positive duration")
    repetitions = min(50, max(1, int(10.0 / first)))
    return statistics.median(measure(*arguments, repetitions) for _ in range(samples))


def _measure_candidates(measurements, arguments: tuple[object, ...], failed) -> tuple[float, ...]:
    def measure(index, repetitions):
        try:
            return measurements[index](*arguments, repetitions)
        except Exception as error:
            failed(index, error)
            raise

    probes = [measure(index, 1) for index in range(len(measurements))]
    for index, elapsed in enumerate(probes):
        if elapsed <= 0:
            failed(index, RuntimeError("native monotonic timing returned a non-positive duration"))
    repetitions = [min(50, max(1, int(10.0 / elapsed))) for elapsed in probes]
    samples = [[] for _ in measurements]
    for round_index in range(3):
        order = list(range(len(measurements)))
        if round_index == 1:
            order.reverse()
        elif round_index == 2:
            order = order[1:] + order[:1]
        for candidate in order:
            samples[candidate].append(measure(candidate, repetitions[candidate]))
    return tuple(statistics.median(values) for values in samples)


@dataclass
class NativeCall(ObservedCall):
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[torch.Tensor, ...]
    key: tuple[object, ...]
    description: tuple
    winner: int | None = None

    def _native_failure(self, error: CompilationStageError) -> None:
        configuration = next((description for candidate, description in zip(
            self.program.candidates, self.program.configuration_descriptions, strict=True)
            if candidate.entry == error.candidate), None)
        details = observation(
            "mojo", self.program.target, self.description, None, (),
            (CandidateObservation(configuration, "failed", error.stage,
                                  type(error).__name__, str(error)),),
            stage="failed", caches=self.program.compilation_cache,
        )
        self._record_observation(details)
        error.observation = details

    def compile(self) -> None:
        """Compile the declared portfolio without loading, choosing or executing it."""
        try:
            self.program.compile()
        except CompilationStageError as error:
            self._native_failure(error)
            raise
        self._record_observation(observation(
            "mojo", self.program.target, self.description, None, (),
            tuple(CandidateObservation(configuration, "compiled", "provider_native_compilation")
                  for configuration in self.program.configuration_descriptions),
            stage="compiled", caches=self.program.compilation_cache,
            history_unavailable="Compilation does not select or time a native candidate",
        ))

    def inspect_configurations(self):
        return self.program.inspect_configurations()

    def _failed_candidate(self, index, error):
        candidate = self.program.candidates[index]
        configuration = self.program.configuration_descriptions[index]
        details = observation("mojo", self.program.target, self.description, configuration, (),
            (CandidateObservation(configuration, "failed", "provider_tuning",
                                  type(error).__name__, str(error)),), stage="failed",
            caches=self.program.compilation_cache)
        self._record_observation(details)
        raise CompilationStageError("provider_tuning", str(error), candidate=candidate.entry,
                                    observation=details) from error

    def choose(self) -> int:
        try:
            self.program.ensure_loaded()
        except CompilationStageError as error:
            self._native_failure(error)
            raise
        if self.winner is not None:
            return self.winner
        reused = self.key in self.program.winners
        observed = ()
        if not reused:
            timings = _measure_candidates(self.program.measurements, self.native_arguments, self._failed_candidate)
            self.program.winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
            self.program.timings[self.key] = timings
            observed = tuple(CandidateObservation(configuration, "trial_completed", "provider_tuning",
                                                  elapsed_ms=elapsed)
                             for configuration, elapsed in zip(self.program.configuration_descriptions, timings, strict=True))
        self.winner = self.program.winners[self.key]
        self._record_observation(observation("mojo", self.program.target, self.description,
            self.program.configuration_descriptions[self.winner], (), observed, stage="selected",
            history_unavailable="The process-local winner was reused; no candidates were retried" if reused else None,
            caches=(*self.program.compilation_cache,
                    CacheObservation("tuning", "process", reused, "Mojo NativeProgram.winners", "selection"))))
        return self.winner

    def launch(self) -> None:
        selected = self.choose()
        try:
            self.program.functions[selected](*self.native_arguments)
        except Exception as error:
            details = replace(self.observation, stage="failed")
            self._record_observation(details)
            raise CompilationStageError("provider_invocation", str(error),
                                        candidate=self.program.candidates[selected].entry, observation=details) from error
        self._record_execution("launched")

    def benchmark(self) -> float:
        selected = self.choose()
        elapsed = _timing_samples(self.program.measurements[selected], self.native_arguments, samples=7)
        self._record_execution("benchmarked")
        return elapsed

    def result(self):
        return invocation_result(self.outputs)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, abi: NativeABI, facts: MojoFacts, target) -> None:
        self._abi = abi
        self._facts = facts
        self._target = target
        self._lock = RLock()
        self._closed = False
        self._loaded_libraries = ()
        self.compilation = None
        self.compilation_cache = ()
        self.functions = ()
        self.measurements = ()
        self.interface = abi.interface
        self.target = bindings({"family": "cpu", **{name: value for name, value in asdict(target.compilation).items()
                                                  if name != "provider"}})
        self.candidates = facts.candidates
        self.configuration_descriptions = tuple(bindings(candidate.metadata()) for candidate in self.candidates)
        self.requirements = requirements = abi.requirements
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(parameter, facts),
            check_view_storage=lambda owner, parameter, facts: requirements.check_storage(parameter, facts),
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.offset, facts.dtype, group),
            scalar_key=lambda parameter, value: parameter.dtype.name,
        )
        self._infer = build_invocation_binders(
            self.interface, observe_view=type(self)._abstract_view,
            allocate_output=type(self)._abstract_output,
            check_relation=check_abstract_relation, abstract=True,
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(
                parameter, facts, check_relation=check_abstract_relation),
        )[0]

    def _check_open(self) -> None:
        if self._closed:
            raise CompilationStageError("provider_native_loading", "Mojo program is closed")

    def compile(self):
        with self._lock:
            self._check_open()
            if self.compilation is None:
                compilation = compile_portfolio(self._facts, self._target, abi=self._abi)
                self.compilation_cache = tuple(CacheObservation(
                    "native_compilation", "persistent_artifact", artifact.cache_hit,
                    "Mojo NativeArtifact.cache_hit", "compilation", artifact.cache_reason,
                    candidate.entry,
                ) for candidate, artifact in zip(self.candidates, compilation.artifacts, strict=True))
                self.winners = _winners.setdefault(compilation.identity, {})
                self.timings = _candidate_timings.setdefault(compilation.identity, {})
                self.compilation = compilation
            return self.compilation

    def ensure_loaded(self) -> None:
        with self._lock:
            compilation = self.compile()
            if self._loaded_libraries:
                return
            libraries, functions, measurements = [], [], []
            argument_types = self._abi.argument_types()
            environment = dict(os.environ)
            try:
                for candidate, artifact in zip(self.candidates, compilation.artifacts, strict=True):
                    if artifact.cache_reason is None and not runtime_inputs_current(artifact, environment):
                        raise RuntimeError("Mojo runtime library resolution changed after native compilation")
                    library = load_native_library(artifact)
                    libraries.append(library)
                    functions.append(library.bind(candidate.entry, argument_types, None))
                    measurements.append(library.bind(
                        candidate.entry + "_benchmark", [*argument_types, ctypes.c_int64], ctypes.c_double))
            except Exception as error:
                for library in libraries:
                    library.close()
                if isinstance(error, CompilationStageError):
                    raise CompilationStageError(
                        error.stage, str(error), cache_directory=error.cache_directory,
                        candidate=candidate.entry, artifacts=error.artifacts,
                    ) from error
                raise CompilationStageError(
                    "provider_native_loading", str(error), cache_directory=artifact.directory,
                    candidate=candidate.entry, artifacts={"library": artifact.library},
                ) from error
            self._loaded_libraries = tuple(libraries)
            self.functions = tuple(functions)
            self.measurements = tuple(measurements)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for library in self._loaded_libraries:
                library.close()
            self._loaded_libraries = ()
            self.functions = ()
            self.measurements = ()

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
        self._check_open()
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs, bound.key,
                          self.describe_arguments(bound, "cpu"))

    @staticmethod
    def _abstract_view(owner, parameter: ViewParameter, value) -> ViewFacts:
        return observe_view(parameter, value, device=torch.device("cpu"), abstract=True)

    @staticmethod
    def _abstract_output(owner, parameter: ViewParameter, shape: tuple):
        return allocate_output(parameter, shape, device=torch.device("cpu"), abstract=True)

    def infer_outputs(self, arguments: tuple):
        return self._infer(self, arguments).result()
