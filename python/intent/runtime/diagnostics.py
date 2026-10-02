"""Observations from an existing provider invocation, never compilation policy."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping


def bindings(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class NativeResource:
    name: str
    value: int | None
    unit: str
    source: str
    stage: str
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    configuration: tuple[tuple[str, object], ...] | None
    status: str
    stage: str
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class InvocationArgument:
    name: str
    dtype: str
    shape: tuple[int, ...] | None
    strides: tuple[int, ...] | None
    device: str | None
    scalar: int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class NativeObservation:
    """Latest actual native invocation; missing SDK facts stay explicitly unknown.

    Arguments describe this invocation, not a promise that every scalar belongs
    to the provider's specialization key. Candidate bindings and native resource
    fields describe the selected compiled program. This is not an occupancy
    estimate, a correctness result, or an Intent optimization decision.
    """

    provider: str
    stage: str
    target: tuple[tuple[str, object], ...]
    invocation: tuple[InvocationArgument, ...]
    configuration: tuple[tuple[str, object], ...] | None
    resources: tuple[NativeResource, ...]
    candidates: tuple[CandidateObservation, ...]
    candidate_history_unavailable: str | None = None
    tuning_cache_hit: bool | None = None


class CandidateRecorder:
    """One bounded record per candidate considered by the existing tuner."""

    def __init__(self) -> None:
        self._records: dict[tuple | None, CandidateObservation] = {}
        self._lock = Lock()

    def record(self, configuration: Mapping[str, object] | None, status: str, stage: str,
               error: Exception | None = None) -> None:
        row = bindings(configuration) if configuration is not None else None
        item = CandidateObservation(row, status, stage,
                                    type(error).__name__ if error is not None else None,
                                    str(error) if error is not None else None)
        with self._lock:
            self._records[row] = item

    def snapshot(self) -> tuple[CandidateObservation, ...]:
        with self._lock:
            return tuple(self._records.values())


def observation(provider: str, target: dict, invocation, configuration,
                resources: tuple[NativeResource, ...], candidates=(), *,
                stage: str = "launched", history_unavailable: str | None = None,
                tuning_cache_hit: bool | None = None) -> NativeObservation:
    arguments = []
    interface = invocation.interface
    for entry in sorted((*interface.public_views, *interface.scalars),
                        key=lambda item: item.parameter.position):
        value = invocation.values[entry.id]
        parameter = entry.parameter
        if entry in interface.public_views:
            arguments.append(InvocationArgument(parameter.name, parameter.dtype.name,
                                                 tuple(value.shape), tuple(value.stride()), str(value.device)))
        else:
            arguments.append(InvocationArgument(parameter.name, parameter.dtype.name, None, None, None, value))
    # These are the emitted target facts, with no local driver object or tensor
    # retained. The actual device is recorded with each view above.
    target_facts = (("family", target["family"]), ("capabilities", bindings(target["capabilities"])))
    selected = None if configuration is None else bindings({
        **configuration, **{name: invocation.values[name] for name in interface.configuration_space.coverage_names}})
    return NativeObservation(provider, stage, target_facts, tuple(arguments), selected,
                             resources, tuple(candidates), history_unavailable, tuning_cache_hit)


def resource(name: str, value: Any, unit: str, source: str, stage: str, *,
             unavailable: str) -> NativeResource:
    if value is None:
        return NativeResource(name, None, unit, source, stage, unavailable)
    if type(value) is not int:
        return NativeResource(name, None, unit, source, stage,
                              f"SDK returned {type(value).__name__}, not an integer resource count")
    return NativeResource(name, value, unit, source, stage)


def unavailable_resources(provider: str, reason: str) -> tuple[NativeResource, ...]:
    return tuple(resource(name, None, unit, provider, "native", unavailable=reason) for name, unit in (
        ("registers_per_thread", "registers"),
        ("local_memory_words_per_thread", "32-bit words"),
        ("shared_memory_bytes", "bytes"),
        ("max_threads_per_block", "threads")))
