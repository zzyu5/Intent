from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from intent.language.annotations import ConstexprSpec
from intent.language.annotations import ViewSpec
from intent.language.dtypes import DType
from intent.language.dtypes import DTypeCategory


class DimExpr:
    __slots__ = ()

    def format(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class StaticDim(DimExpr):
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("static dimension must be an integer")
        if self.value < 0:
            raise ValueError("static dimension must be non-negative")

    def format(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class SymbolDim(DimExpr):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid symbolic dimension: {self.name!r}")

    def format(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class DynamicDim(DimExpr):
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("dynamic dimension label must not be empty")

    def format(self) -> str:
        return f"?{self.label}"


Shape = tuple[DimExpr, ...]


def normalize_dim(dim: int | str | DimExpr) -> DimExpr:
    if isinstance(dim, DimExpr):
        return dim
    if isinstance(dim, bool):
        raise TypeError("boolean is not a shape dimension")
    if isinstance(dim, int):
        return StaticDim(dim)
    if isinstance(dim, str):
        return SymbolDim(dim)
    raise TypeError(f"unsupported shape dimension: {dim!r}")


def normalize_shape(shape: Iterable[int | str | DimExpr]) -> Shape:
    return tuple(normalize_dim(dim) for dim in shape)


class IRType:
    __slots__ = ()

    def format(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True, slots=True)
class ScalarType(IRType):
    dtype: DType

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("scalar type requires an Intent dtype")

    def format(self) -> str:
        return self.dtype.name


@dataclass(frozen=True, slots=True)
class EnumType(IRType):
    name: str
    members: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError("enum type requires an identifier name")
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members:
            raise ValueError("enum type requires at least one member")
        if any(
            not isinstance(member_name, str)
            or not member_name
            or not member_name.isidentifier()
            or isinstance(member_value, bool)
            or not isinstance(member_value, int)
            for member_name, member_value in self.members
        ):
            raise TypeError("enum members must be identifier/integer pairs")
        names = [name for name, _ in self.members]
        values = [value for _, value in self.members]
        if len(set(names)) != len(names) or len(set(values)) != len(values):
            raise ValueError("enum members must have unique names and values")

    def format(self) -> str:
        return f"enum<{self.name}>"


@dataclass(frozen=True, slots=True)
class ConstexprType(IRType):
    value_type: IRType

    def __post_init__(self) -> None:
        if not isinstance(self.value_type, IRType):
            raise TypeError("constexpr value type must be an IRType")

    def format(self) -> str:
        return f"constexpr<{self.value_type}>"


@dataclass(frozen=True, slots=True)
class TensorType(IRType):
    dtype: DType
    shape: Shape

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("tensor type requires an Intent dtype")
        object.__setattr__(self, "shape", normalize_shape(self.shape))

    @property
    def rank(self) -> int:
        return len(self.shape)

    def format(self) -> str:
        dims = "x".join(dim.format() for dim in self.shape)
        return f"tensor<{dims + 'x' if dims else ''}{self.dtype.name}>"


@dataclass(frozen=True, slots=True)
class LogicalIndexType(IRType):
    relation: str

    def __post_init__(self) -> None:
        if not isinstance(self.relation, str) or not self.relation:
            raise ValueError("logical index relation must not be empty")

    def format(self) -> str:
        return f"index<{self.relation}>"


class DomainFlavor(Enum):
    DENSE = "dense"
    STRIDED = "strided"
    PRODUCT = "product"
    RUNTIME = "runtime"
    RAGGED_OUTER = "ragged_outer"
    RAGGED_MEMBER = "ragged_member"


@dataclass(frozen=True, slots=True)
class DomainType(IRType):
    flavor: DomainFlavor = DomainFlavor.DENSE
    rank: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.flavor, DomainFlavor):
            raise TypeError("domain flavor must be a DomainFlavor")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("domain rank must be an integer")
        if self.rank <= 0:
            raise ValueError("domain rank must be positive")

    def format(self) -> str:
        return f"domain<{self.flavor.value},{self.rank}>"


@dataclass(frozen=True, slots=True)
class RegionType(IRType):
    rank: int = 1
    relation: str = "domain"

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError("region rank must be an integer")
        if self.rank <= 0:
            raise ValueError("region rank must be positive")
        if not isinstance(self.relation, str) or not self.relation:
            raise ValueError("region relation must not be empty")

    def format(self) -> str:
        return f"region<{self.relation},{self.rank}>"


class PartitionMode(Enum):
    EXTENT = "extent"
    COUNT = "count"


@dataclass(frozen=True, slots=True)
class PartitionType(IRType):
    mode: PartitionMode
    region_type: RegionType

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PartitionMode):
            raise TypeError("partition mode must be a PartitionMode")
        if not isinstance(self.region_type, RegionType):
            raise TypeError("partition requires a RegionType")

    def format(self) -> str:
        return f"partition<{self.mode.value},{self.region_type.format()}>"


@dataclass(frozen=True, slots=True)
class RaggedType(IRType):
    outer: DomainType
    member: DomainType

    def __post_init__(self) -> None:
        if not isinstance(self.outer, DomainType) or not isinstance(
            self.member, DomainType
        ):
            raise TypeError("ragged descriptor requires outer/member DomainType values")
        if self.outer.flavor is not DomainFlavor.RAGGED_OUTER:
            raise ValueError("ragged outer domain must use ragged_outer flavor")
        if self.member.flavor is not DomainFlavor.RAGGED_MEMBER:
            raise ValueError("ragged member domain must use ragged_member flavor")

    def format(self) -> str:
        return f"ragged<{self.outer.format()},{self.member.format()}>"


@dataclass(frozen=True, slots=True)
class BufferType(IRType):
    dtype: DType
    shape: Shape

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError("buffer type requires an Intent dtype")
        object.__setattr__(self, "shape", normalize_shape(self.shape))

    def format(self) -> str:
        dims = "x".join(dim.format() for dim in self.shape)
        return f"buffer<{dims + 'x' if dims else ''}{self.dtype.name}>"


@dataclass(frozen=True, slots=True)
class RecordType(IRType):
    fields: tuple[tuple[str, IRType], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.fields:
            raise ValueError("record type requires at least one field")
        if any(
            not isinstance(field, tuple)
            or len(field) != 2
            or not isinstance(field[0], str)
            or not isinstance(field[1], IRType)
            for field in self.fields
        ):
            raise TypeError("record fields must be (name, IRType) pairs")
        names = [name for name, _ in self.fields]
        if any(not name or not name.isidentifier() for name in names):
            raise ValueError("record field names must be identifiers")
        if len(set(names)) != len(names):
            raise ValueError("record field names must be unique")

    def format(self) -> str:
        body = ",".join(f"{name}:{field_type}" for name, field_type in self.fields)
        return f"record<{body}>"


def type_from_annotation(annotation: object) -> IRType:
    if isinstance(annotation, ViewSpec):
        return TensorType(annotation.dtype, normalize_shape(annotation.shape))
    if isinstance(annotation, DType):
        return ScalarType(annotation)
    if isinstance(annotation, ConstexprSpec):
        inner = type_from_python_type(annotation.value_type)
        return ConstexprType(inner)
    return type_from_python_type(annotation)


def type_from_python_type(value_type: object) -> IRType:
    from enum import IntEnum

    from intent.language import bool as intent_bool
    from intent.language import f64
    from intent.language import i64

    if value_type is bool:
        return ScalarType(intent_bool)
    if value_type is int:
        return ScalarType(i64)
    if value_type is float:
        return ScalarType(f64)
    if isinstance(value_type, type) and issubclass(value_type, IntEnum):
        return EnumType(
            value_type.__name__,
            tuple((member.name, int(member.value)) for member in value_type),
        )
    raise TypeError(f"unsupported Intent annotation: {value_type!r}")


def is_boolean(value_type: IRType) -> bool:
    if isinstance(value_type, ConstexprType):
        return is_boolean(value_type.value_type)
    return (
        isinstance(value_type, ScalarType)
        and value_type.dtype.category is DTypeCategory.BOOL
    )


def is_integer(value_type: IRType) -> bool:
    if isinstance(value_type, ConstexprType):
        return is_integer(value_type.value_type)
    return isinstance(value_type, (ScalarType, LogicalIndexType)) and (
        isinstance(value_type, LogicalIndexType)
        or value_type.dtype.category
        in (
            DTypeCategory.SIGNED_INTEGER,
            DTypeCategory.UNSIGNED_INTEGER,
            DTypeCategory.INDEX,
        )
    )


def is_numeric(value_type: IRType) -> bool:
    if isinstance(value_type, ConstexprType):
        return is_numeric(value_type.value_type)
    if isinstance(value_type, LogicalIndexType):
        return True
    return isinstance(value_type, ScalarType) and value_type.dtype.category is not DTypeCategory.BOOL


def dims_compatible(lhs: DimExpr, rhs: DimExpr) -> bool:
    if lhs == rhs:
        return True
    if isinstance(lhs, DynamicDim) or isinstance(rhs, DynamicDim):
        return True
    return False


def types_compatible(lhs: IRType, rhs: IRType) -> bool:
    if lhs == rhs:
        return True
    if isinstance(lhs, TensorType) and isinstance(rhs, TensorType):
        return (
            lhs.dtype == rhs.dtype
            and lhs.rank == rhs.rank
            and all(dims_compatible(a, b) for a, b in zip(lhs.shape, rhs.shape))
        )
    if isinstance(lhs, ConstexprType) and isinstance(rhs, ConstexprType):
        return types_compatible(lhs.value_type, rhs.value_type)
    if isinstance(lhs, RecordType) and isinstance(rhs, RecordType):
        return (
            tuple(name for name, _ in lhs.fields)
            == tuple(name for name, _ in rhs.fields)
            and all(
                types_compatible(a, b)
                for (_, a), (_, b) in zip(lhs.fields, rhs.fields)
            )
        )
    return False


def broadcast_shape(lhs: Shape, rhs: Shape) -> Shape:
    result: list[DimExpr] = []
    lhs_rev = list(reversed(lhs))
    rhs_rev = list(reversed(rhs))
    for index in range(max(len(lhs_rev), len(rhs_rev))):
        left = lhs_rev[index] if index < len(lhs_rev) else StaticDim(1)
        right = rhs_rev[index] if index < len(rhs_rev) else StaticDim(1)
        if isinstance(left, StaticDim) and left.value == 1:
            result.append(right)
        elif isinstance(right, StaticDim) and right.value == 1:
            result.append(left)
        elif dims_compatible(left, right):
            if isinstance(left, DynamicDim):
                result.append(left)
            elif isinstance(right, DynamicDim):
                result.append(right)
            else:
                result.append(left)
        else:
            raise ValueError(f"cannot broadcast dimensions {left} and {right}")
    return tuple(reversed(result))


def static_numel(shape: Shape) -> int | None:
    if not all(isinstance(dim, StaticDim) for dim in shape):
        return None
    return math.prod(dim.value for dim in shape if isinstance(dim, StaticDim))
