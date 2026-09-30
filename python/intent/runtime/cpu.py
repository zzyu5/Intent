from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    position: int
    name: str
    dtype: str


@dataclass(frozen=True, slots=True)
class ViewParameter:
    position: int
    name: str
    dtype: str
    shape: tuple[int, ...]
    dimensions: tuple[int, ...]
    access: int
    alias: str
    noalias: bool
    static_axes: tuple[tuple[int, int], ...]
    dimension_axes: tuple[tuple[int, int], ...]

    def bind_dimensions(self, shape: tuple[int, ...], dimensions: dict[int, int]) -> None:
        for axis, extent in self.static_axes:
            if shape[axis] != extent:
                raise ValueError(f"{self.name} has an incompatible static extent")
        for axis, identity in self.dimension_axes:
            extent = shape[axis]
            if identity in dimensions and dimensions[identity] != extent:
                raise ValueError("CPU views disagree on a canonical dimension identity")
            dimensions[identity] = extent

    def output_shape(self, dimensions: dict[int, int]) -> tuple[int, ...]:
        return tuple(static if static >= 0 else dimensions[identity]
                     for static, identity in zip(self.shape, self.dimensions))


@dataclass(slots=True)
class ViewFacts:
    """Observed for one invocation; never retained as facts about a later call."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    pointer: int
    allocation: int
    offset: int
    dtype: object
    begin: int
    end: int


@dataclass(frozen=True, slots=True)
class AliasCheck:
    left: int
    right: int
    writable: bool
    noalias: bool
    same_allocation: bool


@dataclass(slots=True)
class BoundArguments:
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[object, ...]
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class CPUInterface:
    """Immutable host ABI only; implementations and tasks remain compiler-owned."""

    parameters: tuple[ScalarParameter | ViewParameter, ...]
    inputs: tuple[ScalarParameter | ViewParameter, ...]
    outputs: tuple[ViewParameter, ...]
    allocated_outputs: tuple[ViewParameter, ...]
    mutable_inputs: tuple[ViewParameter, ...]
    alias_checks: tuple[AliasCheck, ...]

    @classmethod
    def read(cls, parameters: Sequence[dict[str, object]]) -> CPUInterface:
        parsed: list[ScalarParameter | ViewParameter] = []
        views: list[ViewParameter] = []
        for position, parameter in enumerate(parameters):
            if parameter["kind"] == "scalar":
                parsed.append(ScalarParameter(position, parameter["name"], parameter["dtype"]))
                continue
            if parameter["kind"] != "view":
                raise ValueError("CPU interface parameter must be a view or scalar")
            shape, dimensions = tuple(parameter["shape"]), tuple(parameter["dimensions"])
            view = ViewParameter(
                position, parameter["name"], parameter["dtype"], shape, dimensions,
                parameter["access"], parameter["alias"], parameter["noalias"],
                tuple((axis, extent) for axis, extent in enumerate(shape) if extent >= 0),
                tuple((axis, identity) for axis, identity in enumerate(dimensions) if identity > 0),
            )
            parsed.append(view)
            views.append(view)
        checks = []
        for index, left in enumerate(views):
            for right in views[index + 1:]:
                writable = left.access != 0 or right.access != 0
                noalias = left.noalias or right.noalias
                same_allocation = bool(left.alias and left.alias == right.alias)
                if writable or noalias or same_allocation:
                    checks.append(AliasCheck(left.position, right.position,
                                             writable, noalias, same_allocation))
        return cls(
            tuple(parsed),
            tuple(parameter for parameter in parsed
                  if not isinstance(parameter, ViewParameter) or parameter.access != 1),
            tuple(view for view in views if view.access != 0),
            tuple(view for view in views if view.access == 1),
            tuple(view for view in views if view.access == 2),
            tuple(checks),
        )

    def bind(
        self, arguments: tuple[object, ...], *, explicit_outputs: bool,
        observe_view: Callable[[ViewParameter, object], ViewFacts],
        allocate_output: Callable[[ViewParameter, tuple[int, ...]], object],
        scalar_key_values: bool,
        view_dtype_before_offset: bool,
    ) -> BoundArguments:
        supplied = self.parameters if explicit_outputs else self.inputs
        if len(arguments) != len(supplied):
            raise TypeError(f"expected {len(supplied)} CPU artifact arguments, got {len(arguments)}")
        values: list[object] = [None] * len(self.parameters)
        facts: list[ViewFacts | None] = [None] * len(self.parameters)
        dimensions: dict[int, int] = {}

        def bind(parameter: ScalarParameter | ViewParameter, value: object) -> None:
            values[parameter.position] = value
            if isinstance(parameter, ViewParameter):
                observed = observe_view(parameter, value)
                parameter.bind_dimensions(observed.shape, dimensions)
                facts[parameter.position] = observed

        for parameter, value in zip(supplied, arguments):
            bind(parameter, value)
        if not explicit_outputs:
            for parameter in self.allocated_outputs:
                bind(parameter, allocate_output(parameter, parameter.output_shape(dimensions)))
        for check in self.alias_checks:
            left, right = facts[check.left], facts[check.right]
            same_allocation = left.allocation == right.allocation
            if check.writable and left.begin < right.end and right.begin < left.end:
                raise NotImplementedError("overlapping writable CPU views are not implemented")
            if check.noalias and same_allocation:
                raise ValueError("CPU invocation violates an author noalias constraint")
            if check.same_allocation and not same_allocation:
                raise ValueError("CPU invocation violates an author allocation-alias constraint")
        native: list[object] = []
        key: list[object] = []
        groups: dict[int, int] = {}
        for parameter, value, view in zip(self.parameters, values, facts):
            if view is not None:
                native.extend((view.pointer, *view.shape, *view.strides))
                group = groups.setdefault(view.allocation, len(groups))
                if view_dtype_before_offset:
                    key.append((view.shape, view.strides, view.dtype, view.offset, group))
                else:
                    key.append((view.shape, view.strides, view.offset, view.dtype, group))
            else:
                native.append(value)
                key.append((parameter.dtype, value) if scalar_key_values else parameter.dtype)
        return BoundArguments(tuple(values), tuple(native),
                              tuple(values[parameter.position] for parameter in self.outputs), tuple(key))
