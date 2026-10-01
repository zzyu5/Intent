from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..artifact import ParameterRole, TuningConfiguration, TuningParameter
from .expressions import Expression


class ConfigurationSpace:
    """Final IR candidate bindings and their invocation-dependent constraints.

    Rows retain compiler order. Coverage is bound from invocation facts; provider
    adapters only project a row into their native JIT and tuner interfaces.
    """

    def __init__(self, interface: dict) -> None:
        self.rows = tuple(interface["configurations"])
        self.parameters = tuple(TuningParameter(
            entry["name"], ParameterRole(entry["role"]), entry["category"],
            tuple(entry["candidates"]), entry.get("dimension"),
            tuple(entry["source"]) if entry.get("source") is not None else None,
            tuple(entry["argument_axis"]) if entry.get("argument_axis") is not None else None,
        ) for entry in interface["parameters"])
        self.resource_bounds = tuple((Expression.read(bound["lhs"]), Expression.read(bound["rhs"]))
                                     for bound in interface["resource_bounds"])
        self.coverage = tuple((entry["name"], Expression.read(entry["coverage"]),
                               tuple(entry["candidates"]))
                              for entry in interface["parameters"] if entry.get("coverage") is not None)
        self.coverage_names = tuple(name for name, _, _ in self.coverage)
        self.bound_names = frozenset(parameter.name for parameter in self.parameters
                                     if parameter.name not in self.coverage_names)

    def bind_coverage(self, values: dict[str, object]) -> None:
        for name, bound, candidates in self.coverage:
            required = bound(values)
            selected = next((candidate for candidate in candidates if candidate >= required), None)
            if selected is None:
                raise ValueError(f"no legal full-coverage extent for {name}: required {required}")
            values[name] = selected

    def within_resources(self, values: Mapping[str, object]) -> bool:
        return all(lhs(values) <= rhs(values) for lhs, rhs in self.resource_bounds)

    def candidates(self, values: Mapping[str, object]) -> tuple[dict, ...]:
        result = tuple(row for row in self.rows if self.within_resources({**values, **row}))
        if not result:
            raise ValueError("no configuration satisfies the physical resource bounds")
        return result

    def enumerate(self, values: Mapping[str, object], rows: Iterable[Mapping[str, int]]) -> tuple[TuningConfiguration, ...]:
        result = []
        for row in rows:
            bindings = {**values, **row}
            result.append(TuningConfiguration(self.parameters,
                                               tuple(bindings[parameter.name] for parameter in self.parameters)))
        return tuple(result)
