from __future__ import annotations

from collections.abc import Callable, Sequence
import ctypes
from dataclasses import dataclass

from ..language.dtypes import DType, dtype
from .interface import AliasCheck, PublicInterface, ScalarParameter, ViewAxis, ViewParameter


_CARRIERS = {"ptr": ctypes.c_void_p, "bool": ctypes.c_bool,
             "i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32,
             "i64": ctypes.c_int64, "f32": ctypes.c_float, "f64": ctypes.c_double}

@dataclass(slots=True)
class ViewFacts:
    """Observed for one invocation; never retained as facts about a later call."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    pointer: int
    allocation: int
    allocation_end: int
    offset: int
    dtype: object
    begin: int
    end: int


@dataclass(slots=True)
class BoundArguments:
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[object, ...]
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class NativeSlot:
    parameter: ScalarParameter | ViewParameter
    role: str
    carrier: str
    axis: int | None = None
    element: DType | None = None

    @property
    def name(self) -> str:
        name = f"a{self.parameter.position}"
        return name if self.axis is None else f"{name}_{'d' if self.role == 'extent' else 's'}{self.axis}"


class NativeABI:
    """CPU/DSA pointer, extent and stride slots over the public declaration."""

    def __init__(self, interface: PublicInterface, slots: tuple[NativeSlot, ...]) -> None:
        self.interface = interface
        self.slots = slots
        self.pointer_slots = {slot.parameter.position: slot for slot in slots if slot.role == "pointer"}
        self.alias_checks = interface.alias_checks

    @classmethod
    def read(cls, metadata: dict) -> NativeABI:
        interface = PublicInterface.read(metadata["interface"])
        slots, seen = [], set()
        for entry in metadata["native"]["slots"]:
            position = entry["parameter"]
            if type(position) is not int or not 0 <= position < len(interface.parameters):
                raise ValueError("native slot references an unknown public parameter")
            parameter, role, carrier = interface.parameters[position], entry["role"], entry["carrier"]
            axis, element = None, None
            if ("axis" in entry) != (role in {"extent", "stride"}) or ("element" in entry) != (role == "pointer"):
                raise ValueError("native slot fields disagree with its role")
            if carrier not in _CARRIERS:
                raise NotImplementedError(f"native scalar carrier {carrier!r} is unsupported")
            if role == "scalar":
                if not isinstance(parameter, ScalarParameter) or carrier == "ptr":
                    raise ValueError("native scalar slot requires a scalar public parameter and carrier")
            elif isinstance(parameter, ViewParameter):
                if role == "pointer":
                    if carrier != "ptr":
                        raise ValueError("native pointer slot requires a pointer carrier")
                    element = dtype(entry["element"])
                elif role in {"extent", "stride"}:
                    axis = entry["axis"]
                    if type(axis) is not int or not 0 <= axis < len(parameter.shape) or carrier != "i64":
                        raise ValueError("native metadata slot requires a view axis and signed 64-bit carrier")
                else:
                    raise ValueError("unknown native view slot role")
            else:
                raise ValueError("native view slot requires a public view parameter")
            identity = position, role, axis
            if identity in seen:
                raise ValueError("native ABI has duplicate parameter slots")
            seen.add(identity)
            slots.append(NativeSlot(parameter, role, carrier, axis, element))
        for parameter in interface.parameters:
            required = "pointer" if isinstance(parameter, ViewParameter) else "scalar"
            if (parameter.position, required, None) not in seen:
                raise ValueError("native ABI does not bind every public parameter")
            if isinstance(parameter, ViewParameter):
                for axis in range(len(parameter.shape)):
                    if any((parameter.position, role, axis) not in seen for role in ("extent", "stride")):
                        raise ValueError("native ABI must bind the extent and stride of every view axis")
        return cls(interface, tuple(slots))

    def argument_types(self) -> tuple[object, ...]:
        return tuple(_CARRIERS[slot.carrier] for slot in self.slots)

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
            relations = self.interface.binding_relations(explicit_outputs=explicit_outputs)
            supplied = relations.supplied
            views = {binding.parameter.position: binding for binding in relations.views}
            namespace = {"_observe": observe_view, "_allocate": allocate_output,
                         "_BoundArguments": BoundArguments, "_check_alias": check_alias,
                         "_view_key": view_key, "_scalar_key": scalar_key,
                         "_scalar_argument": scalar_argument, "_check_dimensions": check_dimensions}
            for parameter in self.interface.parameters:
                namespace[f"_p{parameter.position}"] = parameter
            lines = [
                "def bind(program, arguments):",
                f"    if len(arguments) != {len(supplied)}:",
                f"        raise TypeError(f'expected {len(supplied)} native artifact arguments, got {{len(arguments)}}')",
            ]
            def reference(value: int | ViewAxis, field: str) -> str:
                return f"f{value.parameter}.{field}[{value.axis}]" if isinstance(value, ViewAxis) else repr(value)

            def check_view(parameter: ViewParameter, *, allocated: bool = False) -> None:
                binding = views[parameter.position]
                for field, checks in (("shape", binding.shape_checks), ("strides", binding.stride_checks)):
                    for axis, expected in checks:
                        # Fresh dynamic Out extents came from exactly this owner.
                        if allocated and field == "shape" and parameter.shape[axis] < 0:
                            continue
                        lines.append(f"    if f{parameter.position}.{field}[{axis}] != {reference(expected, field)}:")
                        lines.append(f"        raise ValueError({parameter.name + ' violates a declared ' + field + ' relation'!r})")

            for position, parameter in enumerate(supplied):
                index = parameter.position
                lines.append(f"    v{index} = arguments[{position}]")
                if isinstance(parameter, ViewParameter):
                    lines.append(f"    f{index} = _observe(program, _p{index}, v{index})")
                    check_view(parameter)
            if check_dimensions is not None:
                supplied_positions = {parameter.position for parameter in supplied}
                bindings = ", ".join(f"{identity}: {reference(value, 'shape')}" for identity, value in relations.dimensions
                                     if value.parameter in supplied_positions)
                lines.append(f"    _check_dimensions(program, {{{bindings}}})")
            if not explicit_outputs:
                for parameter in self.interface.outputs:
                    extents = []
                    for axis, extent in enumerate(views[parameter.position].output_shape):
                        if extent is None:
                            message = (
                                f"cannot infer {parameter.name} output dimension {parameter.dimensions[axis]} "
                                "from supplied inputs; provide explicit output buffers"
                            )
                            lines.append(f"    raise ValueError({message!r})")
                            break
                        extents.append(reference(extent, "shape"))
                    else:
                        index = parameter.position
                        lines.append(f"    v{index}, f{index} = _allocate(program, _p{index}, {tuple_expression(extents)})")
                        check_view(parameter, allocated=True)
                        continue
                    break
            for index, check in enumerate(self.alias_checks):
                namespace[f"_a{index}"] = check
                left, right = f"f{check.left}", f"f{check.right}"
                lines.append(f"    _check_alias(_a{index}, {left}, {right})")
            native, key = [], []
            if view_key is not None:
                lines.append("    groups = {}")
            for slot in self.slots:
                index = slot.parameter.position
                if slot.role == "pointer":
                    native.append(f"f{index}.pointer")
                elif slot.role == "scalar":
                    native.append(f"v{index}" if scalar_argument is None else f"_scalar_argument(_p{index}, v{index})")
                else:
                    native.append(f"f{index}.{'shape' if slot.role == 'extent' else 'strides'}[{slot.axis}]")
            for parameter in self.interface.parameters:
                index = parameter.position
                value, view = f"v{index}", f"f{index}"
                if isinstance(parameter, ViewParameter):
                    if view_key is not None:
                        lines.append(f"    g{index} = groups.setdefault({view}.allocation, len(groups))")
                        key.append(f"_view_key({view}, g{index})")
                else:
                    if scalar_key is not None:
                        key.append(f"_scalar_key(_p{index}, {value})")
            values = tuple_expression([f"v{parameter.position}" for parameter in self.interface.parameters])
            outputs = tuple_expression([f"v{parameter.position}" for parameter in self.interface.views if parameter.writable])
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

    interface: PublicInterface

    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None):
        if outputs is None:
            return self.prepare(arguments)
        return self.prepare(self.interface.explicit_arguments(arguments, outputs), explicit_outputs=True)
