from __future__ import annotations

import math
from dataclasses import dataclass

from intent.diagnostics import LanguageUseError


@dataclass(frozen=True, slots=True)
class Intrinsic:
    name: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"I.{self.name} is only valid inside an @intent.kernel or @intent.fn body"
        )

    def __repr__(self) -> str:
        return f"I.{self.name}"


@dataclass(frozen=True, slots=True)
class IntrinsicNamespace:
    name: str
    members: tuple[str, ...]

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise LanguageUseError(
            f"I.{self.name} is only valid inside an @intent.kernel or @intent.fn body"
        )

    def __getattr__(self, member: str) -> Intrinsic:
        if member not in self.members:
            raise AttributeError(member)
        return Intrinsic(f"{self.name}.{member}")

    def __repr__(self) -> str:
        return f"I.{self.name}"


auto = Intrinsic("auto")
domain = Intrinsic("domain")
partition = Intrinsic("partition")
parallel = Intrinsic("parallel")
ordered = Intrinsic("ordered")
state_stream = Intrinsic("state_stream")
indices = Intrinsic("indices")
end = Intrinsic("end")
assume_in_bounds = Intrinsic("assume_in_bounds")

reshape = Intrinsic("reshape")
transpose = Intrinsic("transpose")
full = Intrinsic("full")
zeros = Intrinsic("zeros")
record = Intrinsic("record")
cast = Intrinsic("cast")
mask = Intrinsic("mask")

exp = Intrinsic("exp")
exp2 = Intrinsic("exp2")
log = Intrinsic("log")
sin = Intrinsic("sin")
cos = Intrinsic("cos")
floor = Intrinsic("floor")
rsqrt = Intrinsic("rsqrt")
sigmoid = Intrinsic("sigmoid")
maximum = Intrinsic("maximum")
minimum = Intrinsic("minimum")
any = Intrinsic("any")
all = Intrinsic("all")
add = Intrinsic("add")

reduce = IntrinsicNamespace("reduce", ("max", "sum"))
arg_reduce = IntrinsicNamespace("arg_reduce", ("max",))
scan = Intrinsic("scan")
contract = Intrinsic("contract")
scaled_contract = Intrinsic("scaled_contract")
sparse_contract_2to4 = Intrinsic("sparse_contract_2to4")

gather = Intrinsic("gather")
scatter_unique = Intrinsic("scatter_unique")
scatter_reduce = Intrinsic("scatter_reduce")

buffer = Intrinsic("buffer")
store = Intrinsic("store")
mutable_load = Intrinsic("mutable_load")
atomic_add = Intrinsic("atomic_add")
atomic_cas = Intrinsic("atomic_cas")

ragged = Intrinsic("ragged")
members = Intrinsic("members")
random = Intrinsic("random")

inf = math.inf
LOG2E = math.log2(math.e)


INTRINSICS = {
    intrinsic.name: intrinsic
    for intrinsic in (
        auto,
        domain,
        partition,
        parallel,
        ordered,
        state_stream,
        indices,
        end,
        assume_in_bounds,
        reshape,
        transpose,
        full,
        zeros,
        record,
        cast,
        mask,
        exp,
        exp2,
        log,
        sin,
        cos,
        rsqrt,
        sigmoid,
        maximum,
        minimum,
        any,
        all,
        add,
        reduce,
        reduce.max,
        reduce.sum,
        arg_reduce,
        arg_reduce.max,
        scan,
        contract,
        scaled_contract,
        sparse_contract_2to4,
        gather,
        scatter_unique,
        scatter_reduce,
        buffer,
        store,
        mutable_load,
        atomic_add,
        atomic_cas,
        ragged,
        members,
        random,
    )
}
