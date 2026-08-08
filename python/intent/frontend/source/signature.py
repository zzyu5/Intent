from __future__ import annotations

import inspect
from enum import IntEnum
from dataclasses import dataclass

from intent.api import Definition
from intent.api import DefinitionKind
from intent.ir import ConstexprType
from intent.ir import FunctionKind
from intent.ir import IRType
from intent.ir import ParameterKind
from intent.ir import ParameterSpec
from intent.ir import TensorType
from intent.ir import type_from_annotation
from intent.language import ConstexprSpec
from intent.language import DType
from intent.language import ViewSpec

from ..diagnostics.errors import FrontendError
from .unit import SourceUnit


@dataclass(frozen=True, slots=True)
class LoweredSignature:
    function_kind: FunctionKind
    parameters: tuple[ParameterSpec, ...]
    constexpr_values: dict[str, object]


def lower_kernel_signature(
    definition: Definition[object, object],
    source: SourceUnit,
    constexprs: dict[str, object],
) -> LoweredSignature:
    if definition.kind is not DefinitionKind.KERNEL:
        raise FrontendError("frontend entry must be an @intent.kernel", source.location(source.function))
    signature = definition.signature
    parameters: list[ParameterSpec] = []
    values: dict[str, object] = {}
    unknown = set(constexprs) - set(signature.parameters)
    if unknown:
        name = sorted(unknown)[0]
        raise FrontendError(f"unknown constexpr argument {name!r}", source.location(source.function))

    if source.function.args.vararg is not None or source.function.args.kwarg is not None:
        raise FrontendError(
            "kernel entry cannot use *args or **kwargs",
            source.location(source.function),
        )
    ast_parameters = {
        argument.arg: argument
        for argument in (
            *source.function.args.posonlyargs,
            *source.function.args.args,
            *source.function.args.kwonlyargs,
        )
    }
    for name, parameter in signature.parameters.items():
        node = ast_parameters.get(name, source.function)
        location = source.location(node)
        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            raise FrontendError(f"kernel parameter {name!r} requires an Intent annotation", location)
        if isinstance(annotation, ViewSpec):
            if not annotation.shape:
                raise FrontendError(
                    "rank-zero values use runtime scalar parameters, not tensor views",
                    location,
                )
            if parameter.default is not inspect.Signature.empty:
                raise FrontendError("view parameters cannot have Python defaults", location)
            parameters.append(
                ParameterSpec(
                    name=name,
                    type=TensorType(annotation.dtype, annotation.shape),
                    kind=ParameterKind.VIEW,
                    location=location,
                    view_kind=annotation.kind,
                    constraints=annotation.constraints,
                )
            )
            continue
        if isinstance(annotation, DType):
            if parameter.default is not inspect.Signature.empty:
                raise FrontendError("runtime scalar parameters cannot have Python defaults", location)
            parameters.append(
                ParameterSpec(
                    name=name,
                    type=type_from_annotation(annotation),
                    kind=ParameterKind.RUNTIME_SCALAR,
                    location=location,
                )
            )
            continue
        if isinstance(annotation, ConstexprSpec):
            if name in constexprs:
                value = constexprs[name]
            elif parameter.default is not inspect.Signature.empty:
                value = parameter.default
            else:
                raise FrontendError(f"constexpr parameter {name!r} requires a value", location)
            _validate_constexpr_value(name, annotation, value, location)
            values[name] = value
            parameters.append(
                ParameterSpec(
                    name=name,
                    type=ConstexprType(type_from_annotation(annotation).value_type),
                    kind=ParameterKind.CONSTEXPR,
                    location=location,
                )
            )
            continue
        raise FrontendError(f"unsupported kernel annotation for {name!r}", location)

    if signature.return_annotation not in (inspect.Signature.empty, None, type(None)):
        raise FrontendError("kernel entry cannot return a Python/SSA value", source.location(source.function))
    return LoweredSignature(FunctionKind.KERNEL, tuple(parameters), values)


def lower_helper_parameters(
    source: SourceUnit,
    argument_types: tuple[IRType, ...],
) -> tuple[ParameterSpec, ...]:
    if source.function.args.kwonlyargs:
        raise FrontendError(
            "@intent.fn keyword-only parameters are unsupported",
            source.location(source.function),
        )
    if source.function.args.defaults or any(
        value is not None for value in source.function.args.kw_defaults
    ):
        raise FrontendError(
            "@intent.fn parameters cannot use Python defaults",
            source.location(source.function),
        )
    arguments = tuple(
        (*source.function.args.posonlyargs, *source.function.args.args)
    )
    if source.function.args.vararg is not None or source.function.args.kwarg is not None:
        raise FrontendError("@intent.fn does not accept variadic parameters", source.location(source.function))
    if len(arguments) != len(argument_types):
        raise FrontendError("helper argument count does not match call", source.location(source.function))
    return tuple(
        ParameterSpec(
            name=argument.arg,
            type=argument_type,
            kind=ParameterKind.VALUE,
            location=source.location(argument),
        )
        for argument, argument_type in zip(arguments, argument_types)
    )


def _validate_constexpr_value(
    name: str,
    annotation: ConstexprSpec,
    value: object,
    location: object,
) -> None:
    value_type = annotation.value_type
    if not isinstance(value_type, type):
        raise FrontendError(f"constexpr {name!r} annotation must name a Python type", location)
    valid = isinstance(value, value_type)
    if value_type is int and isinstance(value, bool):
        valid = False
    if issubclass(value_type, IntEnum) and type(value) is not value_type:
        valid = False
    if not valid:
        raise FrontendError(
            f"constexpr {name!r} expects {value_type.__name__}, got {type(value).__name__}",
            location,
        )
