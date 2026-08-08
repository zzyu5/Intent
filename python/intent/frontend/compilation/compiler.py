from __future__ import annotations

import re
from dataclasses import dataclass

from intent.api import Definition
from intent.api import HelperDefinition
from intent.api import KernelDefinition
from intent.frontend.mlir import FunctionState
from intent.frontend.mlir import FunctionKind
from intent.frontend.mlir import MlirBuilder
from intent.frontend.mlir import canonicalize_mlir
from intent.frontend.semantics import ValueType

from ..diagnostics.errors import FrontendError
from ..lowering.ast.context import FunctionLowerer
from ..source.signature import lower_helper_parameters
from ..source.signature import lower_kernel_signature
from ..source.unit import SourceUnit


@dataclass(frozen=True, slots=True)
class HelperKey:
    definition: HelperDefinition[object, object]
    argument_types: tuple[ValueType, ...]


class FrontendCompiler:
    def __init__(
        self,
        definition: KernelDefinition[object, object],
        constexprs: dict[str, object],
    ) -> None:
        self.definition = definition
        self.source = SourceUnit.from_definition(definition)
        self.signature = lower_kernel_signature(definition, self.source, constexprs)
        self.builder = MlirBuilder(definition.__name__, self.source.location(self.source.function))
        self.helper_cache: dict[HelperKey, FunctionState] = {}
        self.active_helpers: set[HelperKey] = set()

    def lower(self) -> str:
        function = self.builder.function(
            self.definition.__name__,
            FunctionKind.KERNEL,
            self.signature.parameters,
            (),
            self.source.location(self.source.function),
        )
        lowerer = FunctionLowerer(
            compiler=self,
            definition=self.definition,
            source=self.source,
            function=function,
            constexpr_values=self.signature.constexpr_values,
        )
        lowerer.lower()
        return canonicalize_mlir(self.builder.emit_module())

    def lower_helper(
        self,
        definition: HelperDefinition[object, object],
        argument_types: tuple[ValueType, ...],
        call_location: object,
    ) -> FunctionState:
        key = HelperKey(definition, argument_types)
        cached = self.helper_cache.get(key)
        if cached is not None and key not in self.active_helpers:
            return cached
        if key in self.active_helpers:
            raise FrontendError(
                "recursive @intent.fn requires an explicit recursive IR contract and is unsupported",
                call_location,
            )
        source = SourceUnit.from_definition(definition)
        parameters = lower_helper_parameters(source, argument_types)
        symbol = self._helper_symbol(definition, argument_types)
        function = self.builder.function(
            symbol,
            FunctionKind.HELPER,
            parameters,
            (),
            source.location(source.function),
        )
        self.helper_cache[key] = function
        self.active_helpers.add(key)
        lowerer = FunctionLowerer(
            compiler=self,
            definition=definition,
            source=source,
            function=function,
            constexpr_values={},
        )
        lowerer.lower()
        self.active_helpers.remove(key)
        return function

    def _helper_symbol(
        self,
        definition: Definition[object, object],
        argument_types: tuple[ValueType, ...],
    ) -> str:
        suffix = "__".join(self._sanitize(value_type.format()) for value_type in argument_types)
        base = self._sanitize(definition.__name__)
        return base if not suffix else f"{base}__{suffix}"

    def _sanitize(self, value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", value)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "value"


def lower_to_mlir(
    definition: KernelDefinition[object, object],
    *,
    constexprs: dict[str, object] | None = None,
) -> str:
    if not isinstance(definition, KernelDefinition):
        raise TypeError("lower_to_mlir expects an @intent.kernel definition")
    return FrontendCompiler(definition, dict(constexprs or {})).lower()
