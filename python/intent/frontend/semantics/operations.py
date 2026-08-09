from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OperationKind(Enum):
    CONSTANT = "constant"
    DIM = "dim"

    DOMAIN = "domain"
    DOMAIN_PRODUCT = "domain_product"
    PARTITION = "partition"
    INDICES = "indices"
    REGION_END = "region_end"

    PARALLEL = "parallel"
    ORDERED = "ordered"
    STATE_STREAM = "state_stream"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    YIELD = "yield"
    CONDITION = "condition"
    BREAK = "break"
    CONTINUE = "continue"

    VIEW_LOAD = "view_load"
    VIEW_STORE = "view_store"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    FULL = "full"
    ZEROS = "zeros"
    MAKE_RECORD = "make_record"
    EXTRACT = "extract"

    UNARY = "unary"
    BINARY = "binary"
    COMPARE = "compare"
    SELECT = "select"
    CAST = "cast"
    MASK = "mask"

    REDUCE = "reduce"
    SCAN = "scan"
    CONTRACT = "contract"

    RAGGED = "ragged"
    RAGGED_OUTER = "ragged_outer"
    RAGGED_MEMBER = "ragged_member"
    MEMBERS = "members"

    GATHER = "gather"
    SCATTER_UNIQUE = "scatter_unique"
    SCATTER_REDUCE = "scatter_reduce"

    BUFFER = "buffer"
    BUFFER_LOAD = "buffer_load"
    BUFFER_STORE = "buffer_store"
    ATOMIC_ADD = "atomic_add"
    ATOMIC_CAS = "atomic_cas"
    FENCE = "fence"
    RANDOM = "random"

    CALL = "call"
    RETURN = "return"


TERMINATORS = {
    OperationKind.YIELD,
    OperationKind.CONDITION,
    OperationKind.BREAK,
    OperationKind.CONTINUE,
    OperationKind.RETURN,
}

REGION_OPS = {
    OperationKind.PARALLEL,
    OperationKind.ORDERED,
    OperationKind.STATE_STREAM,
    OperationKind.IF,
    OperationKind.FOR,
    OperationKind.WHILE,
}

STRUCTURED_OPS = {
    OperationKind.REDUCE,
    OperationKind.SCAN,
    OperationKind.CONTRACT,
    OperationKind.PARALLEL,
    OperationKind.ORDERED,
    OperationKind.STATE_STREAM,
}

EFFECTFUL_OPS = {
    OperationKind.VIEW_LOAD,
    OperationKind.VIEW_STORE,
    OperationKind.SCATTER_UNIQUE,
    OperationKind.SCATTER_REDUCE,
    OperationKind.BUFFER_LOAD,
    OperationKind.BUFFER_STORE,
    OperationKind.ATOMIC_ADD,
    OperationKind.ATOMIC_CAS,
    OperationKind.FENCE,
    OperationKind.RANDOM,
}


@dataclass(frozen=True, slots=True)
class AutoExtent:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid auto extent name: {self.name!r}")


class UnaryOperator(Enum):
    NEGATE = "negate"
    NOT = "not"
    EXP = "exp"
    EXP2 = "exp2"
    LOG = "log"
    RSQRT = "rsqrt"


class BinaryOperator(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    TRUE_DIVIDE = "true_divide"
    FLOOR_DIVIDE = "floor_divide"
    REMAINDER = "remainder"
    POWER = "power"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"


class ComparePredicate(Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class AtomicOrdering(Enum):
    RELAXED = "relaxed"
    ACQUIRE = "acquire"
    RELEASE = "release"
    ACQ_REL = "acq_rel"
    SEQ_CST = "seq_cst"


class MemoryScope(Enum):
    WORK_ITEM = "work_item"
    SUBGROUP = "subgroup"
    WORKGROUP = "workgroup"
    DEVICE = "device"
    SYSTEM = "system"


class IndexTermKind(Enum):
    FULL_SLICE = "full_slice"
    NEW_AXIS = "new_axis"
    STATIC_INDEX = "static_index"
    VALUE_INDEX = "value_index"
    REGION_INDEX = "region_index"
    SLICE = "slice"


@dataclass(frozen=True, slots=True)
class IndexTerm:
    kind: IndexTermKind
    operand_positions: tuple[int | None, ...] = ()
    static_values: tuple[int | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operand_positions", tuple(self.operand_positions))
        object.__setattr__(self, "static_values", tuple(self.static_values))
        if any(
            position is not None
            and (isinstance(position, bool) or not isinstance(position, int) or position < 0)
            for position in self.operand_positions
        ):
            raise ValueError("index operand positions must be non-negative")
        if self.kind in (IndexTermKind.FULL_SLICE, IndexTermKind.NEW_AXIS):
            if self.operand_positions or self.static_values:
                raise ValueError(f"{self.kind.value} index term carries no payload")
        elif self.kind is IndexTermKind.STATIC_INDEX:
            if self.operand_positions or len(self.static_values) != 1:
                raise ValueError("static index requires exactly one static value")
            value = self.static_values[0]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("static index must be an integer")
        elif self.kind in (IndexTermKind.VALUE_INDEX, IndexTermKind.REGION_INDEX):
            if (
                len(self.operand_positions) != 1
                or self.operand_positions[0] is None
                or self.static_values
            ):
                raise ValueError(f"{self.kind.value} requires exactly one operand")
        elif self.kind is IndexTermKind.SLICE:
            if len(self.operand_positions) != 3 or len(self.static_values) != 3:
                raise ValueError("slice index must encode start/stop/step slots")
            for operand_position, static_value in zip(
                self.operand_positions, self.static_values
            ):
                if operand_position is not None and static_value is not None:
                    raise ValueError("slice slot cannot be both dynamic and static")
                if static_value is not None and (
                    isinstance(static_value, bool)
                    or not isinstance(static_value, int)
                ):
                    raise TypeError("static slice values must be integers or None")
            if self.static_values[2] == 0:
                raise ValueError("slice step cannot be zero")


@dataclass(frozen=True, slots=True)
class IndexRelation:
    terms: tuple[IndexTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.terms:
            raise ValueError("index relation requires at least one term")
        if any(not isinstance(term, IndexTerm) for term in self.terms):
            raise TypeError("index relation terms must be IndexTerm values")
