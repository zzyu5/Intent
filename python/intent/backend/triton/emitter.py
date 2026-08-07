from __future__ import annotations

import builtins as python_builtins
import linecache

from intent.backend.artifact import CompiledArtifact
from intent.ir import BinaryOperator
from intent.ir import Module
from intent.ir import OpCode
from intent.ir import Operation
from intent.ir import StaticDim
from intent.ir import Value
from intent.ir import walk_operations
from intent.mlir import emit_mlir
from intent.realizer.model import PhysicalPlan
from intent.realizer.model import TailKind
from intent.realizer.verify import verify_plan


def emit_triton(module: Module, plan: PhysicalPlan) -> CompiledArtifact:
    verify_plan(module, plan)
    source = _emit_source(module, plan)
    kernel = _materialize_kernel(source, plan.entry_name)
    launcher = _make_launcher(module, plan, kernel)
    return CompiledArtifact(
        source=source,
        intent_ir=emit_mlir(module),
        plan=plan,
        launch=plan.launch,
        _launcher=launcher,
    )


def _emit_source(module: Module, plan: PhysicalPlan) -> str:
    entry = module.entry()
    reserved = {"N_ELEMENTS", "BLOCK_SIZE"}
    parameter_names = [parameter.spec.name for parameter in entry.parameters]
    if reserved & set(parameter_names):
        raise ValueError("kernel parameter collides with Triton realization metadata")

    operations = {operation.id: operation for operation in walk_operations(entry.body)}
    loop = operations[plan.launch.loop_node_id]
    body = loop.regions[0].blocks[0]
    logical_index = body.arguments[0]
    value_names = {
        parameter.value: parameter.spec.name for parameter in entry.parameters
    }
    value_names[logical_index] = "offsets"
    used_names = set(parameter_names) | {"program", "offsets", "valid"} | reserved

    signature = ", ".join(
        [*parameter_names, "N_ELEMENTS: tl.constexpr", "BLOCK_SIZE: tl.constexpr"]
    )
    lines = [
        "import triton",
        "import triton.language as tl",
        "",
        "",
        "@triton.jit",
        f"def {entry.name}({signature}):",
        "    program = tl.program_id(0)",
        "    offsets = program * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)",
    ]
    boundary = next(
        binding
        for binding in plan.boundaries
        if binding.loop_node_id == plan.launch.loop_node_id
    )
    masked = boundary.tail is TailKind.MASKED
    if masked:
        lines.append("    valid = offsets < N_ELEMENTS")

    for operation in body.operations:
        if operation.opcode is OpCode.CONSTANT:
            result = operation.results[0]
            name = _result_name(result, "constant", used_names)
            value_names[result] = name
            lines.append(f"    {name} = {operation.attributes['value']!r}")
            continue
        if operation.opcode is OpCode.VIEW_LOAD:
            result = operation.results[0]
            source_name = value_names[operation.operands[0]]
            name = _result_name(result, f"{source_name}_value", used_names)
            value_names[result] = name
            suffix = ", mask=valid, other=0.0" if masked else ""
            lines.append(f"    {name} = tl.load({source_name} + offsets{suffix})")
            continue
        if operation.opcode is OpCode.BINARY:
            result = operation.results[0]
            operator = operation.attributes["operator"]
            symbol = {
                BinaryOperator.ADD: "+",
                BinaryOperator.SUBTRACT: "-",
                BinaryOperator.MULTIPLY: "*",
                BinaryOperator.TRUE_DIVIDE: "/",
            }.get(operator)
            if symbol is None:
                raise NotImplementedError("Triton emitter received unsupported binary operator")
            lhs = value_names.get(operation.operands[0])
            rhs = value_names.get(operation.operands[1])
            if lhs is None or rhs is None:
                raise NotImplementedError("Triton binary operand has no realized value")
            name = _result_name(result, f"{operator.value}_value", used_names)
            value_names[result] = name
            lines.append(f"    {name} = {lhs} {symbol} {rhs}")
            continue
        if operation.opcode is OpCode.VIEW_STORE:
            destination = value_names[operation.operands[0]]
            value_index = operation.attributes["value_operand_index"]
            value = value_names.get(operation.operands[value_index])
            if value is None:
                raise NotImplementedError("Triton store value has no realized definition")
            suffix = ", mask=valid" if masked else ""
            lines.append(f"    tl.store({destination} + offsets, {value}{suffix})")
            continue
        if operation.opcode is OpCode.YIELD:
            continue
        raise NotImplementedError(
            f"Triton emitter cannot lower {operation.opcode.value} in pointwise loop"
        )
    return "\n".join(lines) + "\n"


def _result_name(result: Value, fallback: str, used: set[str]) -> str:
    candidate = result.name_hint or fallback
    if not candidate.isidentifier() or candidate in used:
        candidate = f"{fallback}_{result.id}"
    if candidate in used:
        raise ValueError("unable to assign a unique generated Triton value name")
    used.add(candidate)
    return candidate


def _materialize_kernel(source: str, entry_name: str) -> object:
    filename = f"<intent-triton:{entry_name}>"
    source_lines = source.splitlines(keepends=True)
    linecache.cache[filename] = (len(source), None, source_lines, filename)
    namespace: dict[str, object] = {"__name__": f"intent.generated.{entry_name}"}
    code = python_builtins.compile(source, filename, "exec")
    exec(code, namespace)
    kernel = namespace.get(entry_name)
    if kernel is None:
        raise RuntimeError("generated Triton source did not define the target entry")
    return kernel


def _make_launcher(module: Module, plan: PhysicalPlan, kernel: object):
    import torch

    entry = module.entry()
    parameters = entry.parameters
    expected_shape = tuple(
        dimension.value
        for dimension in parameters[0].value.type.shape
        if isinstance(dimension, StaticDim)
    )

    def launch(*arguments: object) -> object:
        if len(arguments) != len(parameters):
            raise TypeError("compiled Intent entry argument count does not match kernel ABI")
        for argument, parameter in zip(arguments, parameters):
            if not isinstance(argument, torch.Tensor):
                raise TypeError("Triton Intent ABI views require torch.Tensor arguments")
            if argument.device.type != "cuda" or argument.device.index != plan.target.device:
                raise ValueError("Triton Intent ABI views must reside on the target CUDA device")
            if argument.dtype != torch.float32 or tuple(argument.shape) != expected_shape:
                raise ValueError("runtime tensor dtype/shape does not match the compiled View ABI")
            if not argument.is_contiguous():
                raise ValueError("runtime tensor violates layout='contiguous' View constraint")
        intervals = [
            (
                argument.data_ptr(),
                argument.data_ptr() + argument.numel() * argument.element_size(),
            )
            for argument in arguments
        ]
        for index, parameter in enumerate(parameters):
            if not parameter.spec.constraints.noalias:
                continue
            start, end = intervals[index]
            for other_index, (other_start, other_end) in enumerate(intervals):
                if other_index != index and max(start, other_start) < min(end, other_end):
                    raise ValueError("runtime tensors violate noalias View constraints")
        with torch.cuda.device(plan.target.device):
            return kernel[plan.launch.grid](
                *arguments,
                N_ELEMENTS=expected_shape[0],
                BLOCK_SIZE=plan.launch.block_size,
                num_warps=plan.launch.num_warps,
                num_stages=plan.pipeline.stages,
            )

    return launch
