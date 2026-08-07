from __future__ import annotations

from intent.ir import BufferType
from intent.ir import ConstexprType
from intent.ir import DomainType
from intent.ir import DynamicDim
from intent.ir import EnumType
from intent.ir import IRType
from intent.ir import LogicalIndexType
from intent.ir import PartitionType
from intent.ir import RaggedType
from intent.ir import RecordType
from intent.ir import RegionType
from intent.ir import ScalarType
from intent.ir import StaticDim
from intent.ir import StreamType
from intent.ir import SymbolDim
from intent.ir import TensorType
from intent.ir import UnitType
from intent.language import DType


_DTYPE_TYPES = {
    "bool": "i1",
    "index": "index",
    "i4": "i4",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "u4": "ui4",
    "u8": "ui8",
    "u16": "ui16",
    "u32": "ui32",
    "u64": "ui64",
    "f8e4m3fn": "f8E4M3FN",
    "f8e5m2": "f8E5M2",
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
}


def emit_type(value_type: IRType) -> str:
    if isinstance(value_type, ScalarType):
        return emit_dtype(value_type.dtype)
    if isinstance(value_type, TensorType):
        dimensions = "x".join(
            str(dimension.value) if isinstance(dimension, StaticDim) else "?"
            for dimension in value_type.shape
        )
        element = emit_dtype(value_type.dtype)
        return f"tensor<{dimensions + 'x' if dimensions else ''}{element}>"
    logical_types = (
        (LogicalIndexType, "logical_index"),
        (DomainType, "domain"),
        (RegionType, "region"),
        (PartitionType, "partition"),
        (RaggedType, "ragged"),
        (BufferType, "buffer"),
        (RecordType, "record"),
        (ConstexprType, "constexpr"),
        (EnumType, "enum"),
        (StreamType, "stream"),
        (UnitType, "unit"),
    )
    for python_type, mnemonic in logical_types:
        if isinstance(value_type, python_type):
            return f"!intent.{mnemonic}<{quote(emit_type_metadata(value_type))}>"
    raise NotImplementedError(
        f"Kernel IR type {type(value_type).__name__} has no Intent MLIR type"
    )


def emit_dtype(dtype: DType) -> str:
    try:
        return _DTYPE_TYPES[dtype.name]
    except KeyError:
        raise ValueError(f"dtype {dtype.name!r} has no MLIR builtin spelling") from None


def emit_view_type(value_type: TensorType, access: str) -> str:
    return f"!intent.view<{emit_type(value_type)}, {quote(access)}>"


def emit_type_metadata(value_type: IRType) -> str:
    if isinstance(value_type, EnumType):
        members = ",".join(f"{name}={value}" for name, value in value_type.members)
        return f"enum<{value_type.name};{members}>"
    return value_type.format()


def emit_shape_metadata(value_type: IRType) -> list[str] | None:
    if not isinstance(value_type, (TensorType, BufferType)):
        return None
    result: list[str] = []
    for dimension in value_type.shape:
        if isinstance(dimension, StaticDim):
            result.append(str(dimension.value))
        elif isinstance(dimension, SymbolDim):
            result.append(dimension.name)
        elif isinstance(dimension, DynamicDim):
            result.append(f"?{dimension.label}")
        else:
            raise TypeError(f"unsupported dimension metadata {dimension!r}")
    return result


def quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'
