from __future__ import annotations

from dataclasses import dataclass

from ..tuning import byte_spans_overlap, view_byte_span
from .configurations import ConfigurationSpace
from .expressions import Expression, evaluate_shape, read_expressions


def torch_dtype(name: str):
    import torch

    names = {
        "index": "int64", "i1": "bool", "bool": "bool",
        "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
        "ui8": "uint8", "ui16": "uint16", "ui32": "uint32", "ui64": "uint64",
        "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
        "f16": "float16", "bf16": "bfloat16", "f32": "float32", "f64": "float64",
        "f8e4m3fn": "float8_e4m3fn", "f8e5m2": "float8_e5m2",
    }
    return getattr(torch, names[name.lower()])


@dataclass(frozen=True, slots=True)
class View:
    abi: int
    name: str
    kernel_name: str
    dtype: str
    access: int
    workspace: bool
    shape: tuple[Expression, ...]
    dimensions: tuple[int, ...]
    alias: str
    noalias: bool

    @property
    def output(self) -> bool:
        return self.access == 1 and not self.workspace

    @property
    def writable(self) -> bool:
        return self.access != 0


@dataclass(frozen=True, slots=True)
class Scalar:
    abi: int
    name: str
    kernel_name: str
    dtype: str


@dataclass(slots=True)
class BoundInvocation:
    interface: GPUInterface
    arguments: tuple
    outputs: tuple
    values: dict[str, object]

    @property
    def views(self) -> tuple:
        return tuple(self.values[view.kernel_name] for view in self.interface.views)

    @property
    def public_views(self) -> tuple:
        return tuple(self.values[view.kernel_name] for view in self.interface.public_views)

    def result(self):
        if not self.outputs:
            return None
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs


class GPUInterface:
    """Invocation facts exported from the final GPU program, with no scheduling policy."""

    def __init__(self, metadata: dict, device: int) -> None:
        self.device = device
        interface = metadata["interface"]
        self.views = tuple(View(
            entry["abi"], entry["name"], entry["kernel_name"], entry["dtype"],
            entry["access"], entry["workspace"], read_expressions(entry["shape"]),
            tuple(entry["dimensions"]), entry["alias"], entry["noalias"],
        ) for entry in interface["views"])
        self.scalars = tuple(Scalar(entry["abi"], entry["name"], entry["kernel_name"],
                                    entry["dtype"]) for entry in interface["scalars"])
        by_abi = {entry.abi: entry for entry in (*self.views, *self.scalars)}
        self.parameters = tuple(by_abi[abi] for abi in interface["public_arguments"])
        self.inputs = tuple(entry for entry in self.parameters
                            if not isinstance(entry, View) or not entry.output)
        self.outputs = tuple(entry for entry in self.parameters
                             if isinstance(entry, View) and entry.output)
        self.public_views = tuple(entry for entry in self.views if not entry.workspace)
        self.workspaces = tuple(entry for entry in self.views if entry.workspace)
        self.metadata = tuple(interface["metadata"])
        self._by_abi = by_abi
        self._raw_views = {entry["abi"]: entry for entry in interface["views"]}
        self._strides = {entry["abi"]: read_expressions(entry["strides"])
                         for entry in interface["views"] if entry["has_strides"]}
        self.grid = read_expressions(interface["grid"])
        self.overlaps = tuple(interface["overlaps"])
        self.configuration_space = ConfigurationSpace(interface)
        self._noalias_pairs = tuple((left, right)
                                   for index, left in enumerate(self.public_views)
                                   for right in self.public_views[index + 1:]
                                   if left.noalias or right.noalias)

    def _metadata_values(self, values: dict[str, object]) -> None:
        for entry in self.metadata:
            view = self._by_abi[entry["source_abi"]]
            value = values.get(view.kernel_name)
            if value is None:
                continue
            axis = entry["source_axis"]
            observed = value.shape[axis] if entry["kind"] == "dimension" else value.stride(axis)
            values[entry["name"]] = observed
            values[entry["kernel_name"]] = observed

    def _check_view(self, entry: View, value, dimensions: dict, *, abstract: bool) -> None:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{entry.name} must be a torch.Tensor")
        expected = torch_dtype(entry.dtype)
        if value.dtype != expected:
            raise TypeError(f"{entry.name} must have dtype {expected}, got {value.dtype}")
        if value.ndim != len(entry.shape):
            raise ValueError(f"{entry.name} must have rank {len(entry.shape)}, got {value.ndim}")
        if value.device != torch.device("cuda", self.device):
            raise ValueError(f"{entry.name} must be on cuda:{self.device}, got {value.device}")
        raw_shape = self._raw_views[entry.abi]["shape"]
        for axis, (dimension, expression) in enumerate(zip(entry.dimensions, raw_shape)):
            actual = value.shape[axis]
            label = f"{entry.name}.shape[{axis}]"
            checks = []
            if expression["kind"] == "constant":
                checks.append((expression["value"], str(expression["value"])))
            if dimension > 0:
                if dimension in dimensions:
                    checks.append(dimensions[dimension])
                else:
                    dimensions[dimension] = (actual, label)
            for required, source in checks:
                message = f"{label} must equal {source}"
                if abstract:
                    torch._check(actual == required, lambda message=message: message)
                elif actual != required:
                    raise ValueError(f"{message}, got {actual} and {required}")

    def bind(self, arguments: tuple, *, outputs: tuple | None = None,
             explicit_outputs: bool = False, abstract: bool = False) -> BoundInvocation:
        import torch

        parameters = self.parameters if explicit_outputs else self.inputs
        if len(arguments) != len(parameters):
            names = ", ".join(parameter.name for parameter in parameters)
            raise TypeError(f"expected {len(parameters)} runtime arguments ({names}), got {len(arguments)}")
        if explicit_outputs and outputs is not None:
            raise TypeError("explicit arguments already contain the output buffers")
        if outputs is not None and len(outputs) != len(self.outputs):
            raise TypeError(f"expected {len(self.outputs)} output buffers, got {len(outputs)}")
        values = {parameter.kernel_name: argument for parameter, argument in zip(parameters, arguments)}
        values.update((parameter.name, argument) for parameter, argument in zip(parameters, arguments)
                      if isinstance(parameter, Scalar))
        dimensions = {}
        for parameter, argument in zip(parameters, arguments):
            if isinstance(parameter, View):
                self._check_view(parameter, argument, dimensions, abstract=abstract)
        self._metadata_values(values)
        if not explicit_outputs:
            for index, parameter in enumerate(self.outputs):
                if outputs is None:
                    shape = []
                    for dimension, expression in zip(parameter.dimensions, parameter.shape):
                        shape.append(dimensions[dimension][0] if dimension > 0 and dimension in dimensions
                                     else expression(values))
                    output = torch.empty(tuple(shape), dtype=torch_dtype(parameter.dtype),
                                         device=torch.device("cuda", self.device))
                else:
                    output = outputs[index]
                self._check_view(parameter, output, dimensions, abstract=abstract)
                values[parameter.kernel_name] = output
                self._metadata_values(values)
        for view in self.public_views:
            if view.abi not in self._strides:
                continue
            actual = values[view.kernel_name]
            for axis, expression in enumerate(self._strides[view.abi]):
                expected = expression(values)
                message = f"{view.name}.stride({axis}) violates the compiled stride requirement"
                if abstract:
                    torch._check(actual.stride(axis) == expected, lambda message=message: message)
                elif actual.stride(axis) != expected:
                    raise ValueError(message)
        if abstract:
            return BoundInvocation(self, tuple(values[entry.kernel_name] for entry in self.parameters),
                                   tuple(values[entry.kernel_name] for entry in self.outputs), values)
        for left, right in self._noalias_pairs:
            lhs, rhs = values[left.kernel_name], values[right.kernel_name]
            lhs_storage, rhs_storage = lhs.untyped_storage(), rhs.untyped_storage()
            lhs_base, rhs_base = lhs_storage.data_ptr(), rhs_storage.data_ptr()
            if byte_spans_overlap((lhs_base, lhs_base + lhs_storage.nbytes()),
                                  (rhs_base, rhs_base + rhs_storage.nbytes())):
                raise ValueError(f"{left.name} and {right.name} violate the declared noalias allocation contract")
        self.configuration_space.bind_coverage(values)
        for workspace in self.workspaces:
            values[workspace.kernel_name] = torch.empty(evaluate_shape(workspace.shape, values),
                                                       dtype=torch_dtype(workspace.dtype),
                                                       device=torch.device("cuda", self.device))
            self._metadata_values(values)
        spans = {}
        for overlap in self.overlaps:
            pair = []
            for abi in (overlap["lhs"], overlap["rhs"]):
                if abi not in spans:
                    spans[abi] = view_byte_span(values[self._by_abi[abi].kernel_name])
                pair.append(spans[abi])
            values[overlap["name"]] = byte_spans_overlap(*pair)
        return BoundInvocation(self, tuple(values[entry.kernel_name] for entry in self.parameters),
                               tuple(values[entry.kernel_name] for entry in self.outputs), values)
