from __future__ import annotations

from collections.abc import Callable, Sequence
import ctypes
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
    stride_constraints: tuple[tuple[int, int], ...]

    def stride_mismatch(self, strides: tuple[int, ...]) -> bool:
        return any(strides[axis] != required for axis, required in self.stride_constraints)

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

    def writable_overlap(self, left: ViewFacts, right: ViewFacts) -> bool:
        return self.writable and left.begin < right.end and right.begin < left.end

    def noalias_violation(self, left: ViewFacts, right: ViewFacts) -> bool:
        return self.noalias and left.allocation == right.allocation

    def allocation_violation(self, left: ViewFacts, right: ViewFacts) -> bool:
        return self.same_allocation and left.allocation != right.allocation


@dataclass(slots=True)
class BoundArguments:
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[object, ...]
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class NativeInterface:
    """Declared view/scalar ABI; no device, allocation, or execution policy.

    CPU and DSA entries flatten each view as pointer, extents, then strides.
    The GPU launch interface has its own resource and parameter representation.
    """

    parameters: tuple[ScalarParameter | ViewParameter, ...]
    inputs: tuple[ScalarParameter | ViewParameter, ...]
    outputs: tuple[ViewParameter, ...]
    allocated_outputs: tuple[ViewParameter, ...]
    mutable_inputs: tuple[ViewParameter, ...]
    alias_checks: tuple[AliasCheck, ...]

    @classmethod
    def read(cls, parameters: Sequence[dict[str, object]]) -> NativeInterface:
        parsed: list[ScalarParameter | ViewParameter] = []
        views: list[ViewParameter] = []
        for position, parameter in enumerate(parameters):
            if parameter["kind"] == "scalar":
                parsed.append(ScalarParameter(position, parameter["name"], parameter["dtype"]))
                continue
            if parameter["kind"] != "view":
                raise ValueError("native interface parameter must be a view or scalar")
            shape, dimensions = tuple(parameter["shape"]), tuple(parameter["dimensions"])
            view = ViewParameter(
                position, parameter["name"], parameter["dtype"], shape, dimensions,
                parameter["access"], parameter["alias"], parameter["noalias"],
                tuple((axis, extent) for axis, extent in enumerate(shape) if extent >= 0),
                tuple((axis, identity) for axis, identity in enumerate(dimensions) if identity > 0),
                tuple((axis, stride) for axis, stride in enumerate(parameter["strides"])
                      if stride is not None) if "strides" in parameter else (),
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

    def argument_types(self, scalar_type: Callable[[str], object]) -> tuple[object, ...]:
        types = []
        for parameter in self.parameters:
            if isinstance(parameter, ViewParameter):
                types.extend((ctypes.c_void_p, *((ctypes.c_int64,) * (2 * len(parameter.shape)))))
            else:
                types.append(scalar_type(parameter.dtype))
        return tuple(types)

    def explicit_arguments(self, arguments: tuple, outputs: tuple) -> tuple:
        """Insert supplied Out buffers at their declared ABI positions."""
        if len(arguments) != len(self.inputs):
            raise TypeError(f"expected {len(self.inputs)} native input arguments, got {len(arguments)}")
        if len(outputs) != len(self.allocated_outputs):
            raise TypeError(f"expected {len(self.allocated_outputs)} explicit output buffers, got {len(outputs)}")
        inputs, supplied_outputs = iter(arguments), iter(outputs)
        return tuple(next(supplied_outputs) if isinstance(parameter, ViewParameter) and parameter.access == 1
                     else next(inputs) for parameter in self.parameters)

    def binders(
        self, *,
        observe_view: Callable[[object, ViewParameter, object], ViewFacts],
        allocate_output: Callable[[object, ViewParameter, tuple[int, ...]], tuple[object, ViewFacts]],
        check_alias: Callable[[AliasCheck, ViewFacts, ViewFacts], None],
        view_key: Callable[[ViewFacts, int], object] | None = None,
        scalar_key: Callable[[ScalarParameter, object], object] | None = None,
        scalar_argument: Callable[[ScalarParameter, object], object] | None = None,
        check_dimensions: Callable[[object, dict[int, int]], None] | None = None,
    ) -> tuple[Callable[[object, tuple[object, ...]], BoundArguments], ...]:
        """Compile the two call modes once; bind fresh invocation facts on use.

        Only ABI positions and schema constants enter the generated statements.
        Observers are unbound methods: the callable neither retains a program
        instance nor caches any tensor, allocation or dimension observation.
        """
        if (view_key is None) != (scalar_key is None):
            raise ValueError("a native tuning key requires both view and scalar components")

        def tuple_expression(entries: Sequence[str]) -> str:
            return "(" + ", ".join(entries) + ("," if entries else "") + ")"

        result = []
        for explicit_outputs in (False, True):
            supplied = self.parameters if explicit_outputs else self.inputs
            namespace = {"_observe": observe_view, "_allocate": allocate_output,
                         "_BoundArguments": BoundArguments, "_check_alias": check_alias,
                         "_view_key": view_key, "_scalar_key": scalar_key,
                         "_scalar_argument": scalar_argument, "_check_dimensions": check_dimensions}
            for parameter in self.parameters:
                namespace[f"_p{parameter.position}"] = parameter
            lines = [
                "def bind(program, arguments):",
                f"    if len(arguments) != {len(supplied)}:",
                f"        raise TypeError(f'expected {len(supplied)} native artifact arguments, got {{len(arguments)}}')",
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
                            lines.append("        raise ValueError('native views disagree on a canonical dimension identity')")
                    else:
                        dimensions[identity] = actual

            for position, parameter in enumerate(supplied):
                index = parameter.position
                lines.append(f"    v{index} = arguments[{position}]")
                if isinstance(parameter, ViewParameter):
                    lines.append(f"    f{index} = _observe(program, _p{index}, v{index})")
                    bind_dimensions(parameter)
                    for axis, stride in parameter.stride_constraints:
                        lines.append(f"    if f{index}.strides[{axis}] != {stride}:")
                        lines.append(f"        raise ValueError({parameter.name + ' violates an author stride constraint'!r})")
            if check_dimensions is not None:
                bindings = ", ".join(f"{identity}: {value}" for identity, value in dimensions.items())
                lines.append(f"    _check_dimensions(program, {{{bindings}}})")
            if not explicit_outputs:
                for parameter in self.allocated_outputs:
                    extents = []
                    for extent, identity in zip(parameter.shape, parameter.dimensions):
                        if extent >= 0:
                            extents.append(repr(extent))
                        elif identity in dimensions:
                            extents.append(dimensions[identity])
                        else:
                            message = (
                                f"cannot infer {parameter.name} output dimension {identity} "
                                "from supplied inputs; provide explicit output buffers"
                            )
                            lines.append(f"    raise ValueError({message!r})")
                            break
                    else:
                        index = parameter.position
                        lines.append(f"    v{index}, f{index} = _allocate(program, _p{index}, {tuple_expression(extents)})")
                        bind_dimensions(parameter, allocated=True)
                        continue
                    break
            for index, check in enumerate(self.alias_checks):
                namespace[f"_a{index}"] = check
                left, right = f"f{check.left}", f"f{check.right}"
                lines.append(f"    _check_alias(_a{index}, {left}, {right})")
            native, key = [], []
            if view_key is not None:
                lines.append("    groups = {}")
            for parameter in self.parameters:
                index = parameter.position
                value, view = f"v{index}", f"f{index}"
                if isinstance(parameter, ViewParameter):
                    native.append(f"{view}.pointer")
                    native.extend(f"{view}.shape[{axis}]" for axis in range(len(parameter.shape)))
                    native.extend(f"{view}.strides[{axis}]" for axis in range(len(parameter.shape)))
                    if view_key is not None:
                        lines.append(f"    g{index} = groups.setdefault({view}.allocation, len(groups))")
                        key.append(f"_view_key({view}, g{index})")
                else:
                    native.append(value if scalar_argument is None else f"_scalar_argument(_p{index}, {value})")
                    if scalar_key is not None:
                        key.append(f"_scalar_key(_p{index}, {value})")
            values = tuple_expression([f"v{parameter.position}" for parameter in self.parameters])
            outputs = tuple_expression([f"v{parameter.position}" for parameter in self.outputs])
            lines.append(f"    return _BoundArguments({values}, {tuple_expression(native)}, {outputs}, {tuple_expression(key)})")
            code = compile("\n".join(lines) + "\n", "<intent.native.abi>", "exec", dont_inherit=True)
            exec(code, namespace)
            result.append(namespace["bind"])
        return tuple(result)


class NativePreparedRuntime:
    """Public preparation capability delegating to a native runtime's binder.

    The existing tuple-based prepare method remains the provider entry for
    run/launch and native benchmark callers. Neither preparation path executes.
    """

    interface: NativeInterface

    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None):
        if outputs is None:
            return self.prepare(arguments)
        return self.prepare(self.interface.explicit_arguments(arguments, outputs), explicit_outputs=True)
