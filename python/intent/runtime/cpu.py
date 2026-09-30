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

    def binders(
        self, *,
        observe_view: Callable[[object, ViewParameter, object], ViewFacts],
        allocate_output: Callable[[object, ViewParameter, tuple[int, ...]], tuple[object, ViewFacts]],
        scalar_key_values: bool,
        view_dtype_before_offset: bool,
    ) -> tuple[Callable[[object, tuple[object, ...]], BoundArguments], ...]:
        """Compile the two call modes once; bind fresh invocation facts on use.

        Only ABI positions and schema constants enter the generated statements.
        Observers are unbound methods: the callable neither retains a program
        instance nor caches any tensor, allocation or dimension observation.
        """
        def tuple_expression(entries: Sequence[str]) -> str:
            return "(" + ", ".join(entries) + ("," if entries else "") + ")"

        result = []
        for explicit_outputs in (False, True):
            supplied = self.parameters if explicit_outputs else self.inputs
            namespace = {"_observe": observe_view, "_allocate": allocate_output,
                         "_BoundArguments": BoundArguments}
            for parameter in self.parameters:
                namespace[f"_p{parameter.position}"] = parameter
            lines = [
                "def bind(program, arguments):",
                f"    if len(arguments) != {len(supplied)}:",
                f"        raise TypeError(f'expected {len(supplied)} CPU artifact arguments, got {{len(arguments)}}')",
            ]
            # Dimension owners are ABI relations, not per-call hash entries.
            # Preserve the existing order: supplied views, then allocated Outs.
            dimensions: dict[int, str] = {}

            def bind_dimensions(parameter: ViewParameter, *, allocated: bool = False) -> None:
                view = f"f{parameter.position}"
                if not allocated:
                    for axis, extent in parameter.static_axes:
                        lines.append(f"    if {view}.shape[{axis}] != {extent}:")
                        lines.append(f"        raise ValueError({parameter.name + ' has an incompatible static extent'!r})")
                for axis, identity in parameter.dimension_axes:
                    actual = f"{view}.shape[{axis}]"
                    if identity in dimensions:
                        # A dynamic Out extent was allocated from this exact
                        # owner. Static Out extents still need equality checks.
                        if not allocated or parameter.shape[axis] >= 0:
                            lines.append(f"    if {actual} != {dimensions[identity]}:")
                            lines.append("        raise ValueError('CPU views disagree on a canonical dimension identity')")
                    else:
                        dimensions[identity] = actual

            for position, parameter in enumerate(supplied):
                index = parameter.position
                lines.append(f"    v{index} = arguments[{position}]")
                if isinstance(parameter, ViewParameter):
                    lines.append(f"    f{index} = _observe(program, _p{index}, v{index})")
                    bind_dimensions(parameter)
            if not explicit_outputs:
                for parameter in self.allocated_outputs:
                    extents = []
                    for extent, identity in zip(parameter.shape, parameter.dimensions):
                        if extent >= 0:
                            extents.append(repr(extent))
                        elif identity in dimensions:
                            extents.append(dimensions[identity])
                        else:
                            # Match the existing unsupported implicit-output
                            # shape boundary; explicit output binding still works.
                            lines.append(f"    raise KeyError({identity!r})")
                            break
                    else:
                        index = parameter.position
                        lines.append(f"    v{index}, f{index} = _allocate(program, _p{index}, {tuple_expression(extents)})")
                        bind_dimensions(parameter, allocated=True)
                        continue
                    break
            for check in self.alias_checks:
                left, right = f"f{check.left}", f"f{check.right}"
                if check.writable:
                    lines.append(f"    if {left}.begin < {right}.end and {right}.begin < {left}.end:")
                    lines.append("        raise NotImplementedError('overlapping writable CPU views are not implemented')")
                if check.noalias:
                    lines.append(f"    if {left}.allocation == {right}.allocation:")
                    lines.append("        raise ValueError('CPU invocation violates an author noalias constraint')")
                if check.same_allocation:
                    lines.append(f"    if {left}.allocation != {right}.allocation:")
                    lines.append("        raise ValueError('CPU invocation violates an author allocation-alias constraint')")
            native, key = [], []
            lines.append("    groups = {}")
            for parameter in self.parameters:
                index = parameter.position
                value, view = f"v{index}", f"f{index}"
                if isinstance(parameter, ViewParameter):
                    native.append(f"{view}.pointer")
                    native.extend(f"{view}.shape[{axis}]" for axis in range(len(parameter.shape)))
                    native.extend(f"{view}.strides[{axis}]" for axis in range(len(parameter.shape)))
                    lines.append(f"    g{index} = groups.setdefault({view}.allocation, len(groups))")
                    fields = [f"{view}.shape", f"{view}.strides"]
                    fields.extend((f"{view}.dtype", f"{view}.offset") if view_dtype_before_offset
                                  else (f"{view}.offset", f"{view}.dtype"))
                    key.append(tuple_expression([*fields, f"g{index}"]))
                else:
                    native.append(value)
                    key.append(tuple_expression([repr(parameter.dtype), value])
                               if scalar_key_values else repr(parameter.dtype))
            values = tuple_expression([f"v{parameter.position}" for parameter in self.parameters])
            outputs = tuple_expression([f"v{parameter.position}" for parameter in self.outputs])
            lines.append(f"    return _BoundArguments({values}, {tuple_expression(native)}, {outputs}, {tuple_expression(key)})")
            code = compile("\n".join(lines) + "\n", "<intent.cpu.abi>", "exec", dont_inherit=True)
            exec(code, namespace)
            result.append(namespace["bind"])
        return tuple(result)
