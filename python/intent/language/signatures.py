from __future__ import annotations

from inspect import Parameter, Signature


def _signature(
    positional: tuple[str, ...],
    *,
    required: tuple[str, ...] = (),
    defaults: tuple[tuple[str, object], ...] = (),
) -> Signature:
    return Signature(
        [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in positional]
        + [Parameter(name, Parameter.KEYWORD_ONLY) for name in required]
        + [
            Parameter(name, Parameter.KEYWORD_ONLY, default=value)
            for name, value in defaults
        ]
    )


INTRINSIC_SIGNATURES = {
    "fdiv": _signature(
        ("lhs", "rhs"), defaults=(("approximate", False), ("flush_to_zero", False))
    ),
    "exp2": _signature(
        ("value",), defaults=(("approximate", False), ("flush_to_zero", False))
    ),
    "tanh": _signature(("value",), defaults=(("approximate", False),)),
    "dot": _signature(("lhs", "rhs"), required=("acc_dtype",)),
    "matvec": _signature(
        ("matrix", "vector"), required=("acc_dtype",), defaults=(("transpose", False),)
    ),
    "vecmat": _signature(
        ("vector", "matrix"), required=("acc_dtype",), defaults=(("transpose", False),)
    ),
    "matmul": _signature(
        ("lhs", "rhs"),
        required=("acc_dtype",),
        defaults=(("transpose_lhs", False), ("transpose_rhs", False)),
    ),
    "outer": _signature(("lhs", "rhs")),
    "reduce.sum": _signature(
        ("value",), required=("axis",), defaults=(("acc_dtype", None),)
    ),
    "reduce.max": _signature(
        ("value",), required=("axis",), defaults=(("acc_dtype", None),)
    ),
    "reduce.any": _signature(("value",), required=("axis",)),
    "reduce.all": _signature(("value",), required=("axis",)),
    "cumsum": _signature(
        ("value",),
        required=("axis",),
        defaults=(("inclusive", True), ("reverse", False), ("acc_dtype", None)),
    ),
    "cummax": _signature(
        ("value",),
        required=("axis",),
        defaults=(("inclusive", True), ("reverse", False), ("acc_dtype", None)),
    ),
    "scaled_matmul": _signature(
        ("lhs", "lhs_scale", "rhs", "rhs_scale"),
        required=("lhs_format", "rhs_format", "group_size", "acc_dtype"),
    ),
    "sparse_matmul": _signature(
        ("compressed", "metadata", "rhs"), required=("format", "acc_dtype")
    ),
}
