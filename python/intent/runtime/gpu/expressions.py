from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "floor_div": "//"}


def _source(expression: dict) -> str:
    kind = expression["kind"]
    if kind == "constant":
        value = expression["value"]
        if type(value) is not int:
            raise TypeError("physical constant must be an integer")
        return repr(value)
    if kind in {"dimension", "parameter", "scalar"}:
        return f"values[{expression['symbol']!r}]"
    operands = tuple(_source(operand) for operand in expression["operands"])
    if kind in _BINARY:
        left, right = operands
        return f"({left} {_BINARY[kind]} {right})"
    if kind == "ceil_div":
        left, right = operands
        return f"(-(-{left} // {right}))"
    if kind in {"min", "max"}:
        left, right = operands
        return f"{kind}({left}, {right})"
    if kind == "select":
        condition, yes, no = operands
        return f"({yes} if {condition} else {no})"
    if kind == "next_power_of_two":
        (value,) = operands
        return f"(1 << (max({value}, 1) - 1).bit_length())"
    raise NotImplementedError(f"unknown physical expression kind: {kind}")


@dataclass(frozen=True, slots=True)
class Expression:
    """A compiler expression bound once, with live invocation values on use."""

    evaluate: Callable[[Mapping[str, object]], object]

    @classmethod
    def read(cls, expression: dict) -> Expression:
        source = _source(expression)
        function = eval(compile(f"lambda values: {source}", "<intent physical expression>", "eval"),
                        {"__builtins__": {}, "min": min, "max": max})
        return cls(function)

    def __call__(self, values: Mapping[str, object]):
        return self.evaluate(values)


def read_expressions(expressions: list[dict]) -> tuple[Expression, ...]:
    return tuple(Expression.read(expression) for expression in expressions)


def evaluate_shape(expressions: tuple[Expression, ...], values: Mapping[str, object]) -> tuple:
    return tuple(expression(values) for expression in expressions)
