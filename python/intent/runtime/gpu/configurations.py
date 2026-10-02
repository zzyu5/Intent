from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..artifact import ParameterRole, TuningConfiguration, TuningParameter
from .expressions import Expression


@dataclass(frozen=True, slots=True)
class RequirementEvaluation:
    kind: str
    metric: str
    message: str
    status: str
    usage: int | None
    limit: int | None
    detail: str = ""

    def describe(self) -> str:
        quantity = f"usage={self.usage}, limit={self.limit}" if self.usage is not None and self.limit is not None else self.detail
        return f"{self.kind}/{self.metric}: {self.message} ({self.status}; {quantity})"


@dataclass(frozen=True, slots=True)
class ConfigurationRequirement:
    kind: str
    metric: str
    usage: Expression
    limit: Expression
    message: str

    @classmethod
    def read(cls, entry: dict) -> ConfigurationRequirement:
        kind, metric, message = entry["kind"], entry["metric"], entry["message"]
        if kind not in {"legality", "nominal_budget"}:
            raise ValueError("unknown configuration requirement kind")
        if metric not in {"fragment_elements", "fragment_register_words"}:
            raise ValueError("unknown configuration requirement metric")
        if not isinstance(message, str) or not message:
            raise ValueError("configuration requirement must carry its diagnostic message")
        return cls(kind, metric, Expression.read(entry["usage"]), Expression.read(entry["limit"]), message)

    def assess(self, values: Mapping[int | str, object]) -> RequirementEvaluation:
        missing = (self.usage.references | self.limit.references) - values.keys()
        if missing:
            return RequirementEvaluation(self.kind, self.metric, self.message, "unresolved", None, None,
                                         f"missing bindings {sorted(missing, key=str)}")
        try:
            usage, limit = self.usage(values), self.limit(values)
        except ArithmeticError as error:
            return RequirementEvaluation(self.kind, self.metric, self.message, "unresolved", None, None,
                                         f"{type(error).__name__}: {error}")
        if type(usage) is not int or type(limit) is not int:
            raise TypeError("configuration requirement quantities must evaluate to integers")
        status = "invalid" if usage <= 0 or limit < 0 else "satisfied" if usage <= limit else "exceeded"
        return RequirementEvaluation(self.kind, self.metric, self.message, status, usage, limit)


@dataclass(frozen=True, slots=True)
class CoverageBinding:
    name: str
    bound: Expression
    candidates: tuple[int, ...]

    def select(self, values: Mapping[int | str, object]) -> int:
        required = self.bound(values)
        selected = next((candidate for candidate in self.candidates if candidate >= required), None)
        if selected is None:
            raise ValueError(f"no legal full-coverage extent for {self.name}: required {required}")
        return selected


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
        self.requirements = tuple(ConfigurationRequirement.read(entry) for entry in interface["requirements"])
        self.coverage = tuple(CoverageBinding(entry["name"], Expression.read(entry["coverage"]),
                                               tuple(entry["candidates"]))
                              for entry in interface["parameters"] if entry.get("coverage") is not None)
        self.coverage_names = tuple(binding.name for binding in self.coverage)
        self.bound_names = frozenset(parameter.name for parameter in self.parameters
                                     if parameter.name not in self.coverage_names)
        names = {parameter.name for parameter in self.parameters}
        if len(names) != len(self.parameters) or not self.rows:
            raise ValueError("configuration space requires unique parameters and nonempty candidate rows")
        domains = {parameter.name: parameter.candidates for parameter in self.parameters}
        for row in self.rows:
            if set(row) != self.bound_names or any(type(value) is not int or value not in domains[name]
                                                   for name, value in row.items()):
                raise ValueError("configuration row must bind each non-deferred parameter within its declared domain")
        for requirement in self.requirements:
            references = requirement.usage.references | requirement.limit.references
            if any(isinstance(reference, str) and reference not in names for reference in references):
                raise ValueError("configuration requirement references an undeclared parameter")

    def assess(self, values: Mapping[int | str, object]) -> tuple[RequirementEvaluation, ...]:
        """Evaluate declared candidate conditions, independently of native allocation."""
        return tuple(requirement.assess(values) for requirement in self.requirements)

    @staticmethod
    def _within(evaluations: tuple[RequirementEvaluation, ...]) -> bool:
        unresolved = tuple(entry for entry in evaluations if entry.status == "unresolved")
        if unresolved:
            from ...compiler.toolchain import CompilationStageError
            raise CompilationStageError("candidate_binding", "; ".join(entry.describe() for entry in unresolved))
        return all(entry.status == "satisfied" for entry in evaluations)

    def within_resources(self, values: Mapping[int | str, object]) -> bool:
        return self._within(self.assess(values))

    def candidates(self, values: Mapping[int | str, object]) -> tuple[dict, ...]:
        assessed = tuple((row, self.assess({**values, **row})) for row in self.rows)
        result = tuple(row for row, evaluations in assessed if self._within(evaluations))
        if not result:
            from ...compiler.toolchain import CompilationStageError
            reasons = []
            for row, evaluations in assessed[:4]:
                rejected = [entry.describe() for entry in evaluations if entry.status != "satisfied"]
                reasons.append(f"{row}: " + "; ".join(rejected[:3]))
            if len(assessed) > 4:
                reasons.append(f"{len(assessed) - 4} additional candidate rows rejected")
            raise CompilationStageError("candidate_selection",
                "no candidate satisfies the declared configuration requirements\n" + "\n".join(reasons))
        return result

    def enumerate(self, values: Mapping[int | str, object], rows: Iterable[Mapping[str, int]]) -> tuple[TuningConfiguration, ...]:
        result = []
        for row in rows:
            bindings = {**values, **row}
            result.append(TuningConfiguration(self.parameters,
                                               tuple(bindings[parameter.name] for parameter in self.parameters)))
        return tuple(result)
