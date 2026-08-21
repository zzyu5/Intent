from __future__ import annotations

import ast
from typing import cast

from intent.api import DefinitionKind
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import DomainFlavor
from intent.frontend.semantics import DomainType
from intent.frontend.semantics import Effect
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import PartitionMode
from intent.frontend.semantics import PartitionType
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.language import bool as intent_bool
from intent.language import DType
from intent.language import f64
from intent.language import i64

from .expressions import compile_time_value
from .indexing import lower_index
from .indexing import validate_indexed_value
from .model import ConstexprBinding
from .model import Expression
from .model import IterationSpec
from .model import Literal
from .model import LoopContext
from .model import ShapeValue
from .model import StaticTuple
from .model import StreamSpec


def lower_statement(lowerer: object, node: ast.stmt) -> None:
    if isinstance(node, ast.Assign):
        _lower_assign(lowerer, node)
        return
    if isinstance(node, ast.AnnAssign):
        _lower_annotated_assign(lowerer, node)
        return
    if isinstance(node, ast.AugAssign):
        _lower_augmented_assign(lowerer, node)
        return
    if isinstance(node, ast.Expr):
        operation_count = lowerer.current_block.operation_count
        result = lowerer.lower_expression(node.value)
        if isinstance(result, StaticTuple) and not result.elements:
            return
        if lowerer.current_block.has_effect_since(operation_count):
            return
        lowerer.error(node, "unused pure expression is not a valid Intent statement")
    if isinstance(node, ast.Return):
        _lower_return(lowerer, node)
        return
    if isinstance(node, ast.If):
        _lower_if(lowerer, node)
        return
    if isinstance(node, ast.For):
        _lower_for(lowerer, node)
        return
    if isinstance(node, ast.While):
        _lower_while(lowerer, node)
        return
    if isinstance(node, ast.With):
        _lower_with(lowerer, node)
        return
    if isinstance(node, ast.Break):
        _reject_unlowered_loop_exit(lowerer, node)
        return
    if isinstance(node, ast.Continue):
        _reject_unlowered_loop_exit(lowerer, node)
        return
    if isinstance(node, ast.Assert):
        _lower_assert(lowerer, node)
        return
    if isinstance(node, ast.Pass):
        return
    lowerer.error(node, f"unsupported Python statement {type(node).__name__}")


def _reject_unlowered_loop_exit(lowerer: object, node: ast.Break | ast.Continue) -> None:
    keyword = "break" if isinstance(node, ast.Break) else "continue"
    if not lowerer.loop_stack:
        lowerer.error(node, f"{keyword} requires an enclosing loop")
    context = lowerer.loop_stack[-1]
    if context.opcode not in (OperationKind.FOR, OperationKind.WHILE):
        lowerer.error(
            node,
            f"{keyword} is supported only by ordinary for/while loops",
        )
    lowerer.error(
        node,
        f"{keyword} escaped ordinary-loop control normalization",
    )


def _lower_assign(lowerer: object, node: ast.Assign) -> None:
    expression = lowerer.lower_expression(node.value)
    for target in node.targets:
        _assign_target(lowerer, target, expression)


def _lower_annotated_assign(lowerer: object, node: ast.AnnAssign) -> None:
    if node.value is None:
        lowerer.error(node, "local annotations must initialize an SSA value")
    expression = lowerer.lower_expression(node.value)
    annotation = lowerer.lower_expression(node.annotation)
    if isinstance(annotation, DType) and isinstance(expression, Literal):
        expression = lowerer.materialize(expression, node.value, ScalarType(annotation))
    _assign_target(lowerer, node.target, expression)


def _lower_augmented_assign(lowerer: object, node: ast.AugAssign) -> None:
    synthetic = ast.BinOp(left=node.target, op=node.op, right=node.value)
    ast.copy_location(synthetic, node)
    expression = lowerer.lower_expression(synthetic)
    _assign_target(lowerer, node.target, expression)


def _assign_target(lowerer: object, target: ast.AST, expression: Expression) -> None:
    if isinstance(target, ast.Name):
        if target.id == "_":
            return
        if isinstance(expression, MlirValue) and expression.name_hint is None:
            expression.name_hint = target.id
        lowerer.environment[target.id] = expression
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        elements = _destructure(lowerer, expression, target)
        if len(elements) != len(target.elts):
            lowerer.error(target, "assignment target/value arity mismatch")
        for child, element in zip(target.elts, elements):
            _assign_target(lowerer, child, element)
        return
    if isinstance(target, ast.Subscript):
        _store_subscript(lowerer, target, expression)
        return
    lowerer.error(target, "assignment target must be name, tuple/list, or mutable subscript")


def _destructure(lowerer: object, expression: Expression, node: ast.AST) -> tuple[Expression, ...]:
    if isinstance(expression, StaticTuple):
        return expression.elements
    if isinstance(expression, ShapeValue):
        return tuple(expression.dimensions)
    lowerer.error(node, "value is not destructurable")


def _store_subscript(lowerer: object, target_node: ast.Subscript, expression: Expression) -> None:
    target = lowerer.materialize(lowerer.lower_expression(target_node.value), target_node.value)
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(target_node, "subscript assignment target must be a view or logical buffer")
    if isinstance(target.type, TensorType) and target not in lowerer.view_kinds:
        lowerer.error(target_node, "pure tensor SSA cannot be mutated")
    if isinstance(target.type, TensorType):
        lowerer.require_writable_view(target, target_node)
    lowered = lower_index(lowerer, target, target_node.slice, first_operand_position=2)
    value = lowerer.materialize(
        expression,
        target_node,
        ScalarType(target.type.dtype) if isinstance(expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer,
        value,
        lowered.result_shape,
        target_node,
        expected_dtype=target.type.dtype,
    )
    operands = (target, value, *lowered.operands)
    attributes = {"index": lowered.relation, "value_operand_index": 1}
    if isinstance(target.type, BufferType):
        opcode = OperationKind.BUFFER_STORE
        resource = ResourceKind.LOGICAL_BUFFER
    else:
        opcode = OperationKind.VIEW_STORE
        resource = ResourceKind.EXTERNAL_VIEW
    lowerer.emit(
        opcode,
        lowerer.location(target_node),
        operands=operands,
        attributes=attributes,
        effects=(Effect(EffectKind.WRITE, resource, target),),
    )


def _lower_return(lowerer: object, node: ast.Return) -> None:
    if lowerer.inline_helpers:
        frame = lowerer.inline_helpers[-1]
        if lowerer.current_block is not frame.entry_block:
            lowerer.error(
                node,
                "return inside structured control flow is unsupported; return after the region",
            )
        if node.value is None:
            values: tuple[MlirValue, ...] = ()
        else:
            expression = lowerer.lower_expression(node.value)
            if isinstance(expression, StaticTuple):
                values = tuple(
                    lowerer.materialize(element, node.value)
                    for element in expression.elements
                )
            else:
                values = (lowerer.materialize(expression, node.value),)
        frame.returned = values
        return
    if lowerer.current_block.owner is not lowerer.function.body:
        lowerer.error(node, "return inside structured control flow is unsupported; return after the region")
    if lowerer.definition.kind is DefinitionKind.KERNEL:
        if node.value is not None:
            lowerer.error(node, "kernel returns through Out/InOut views")
        lowerer.emit(OperationKind.RETURN, lowerer.location(node))
        return
    lowerer.error(node, "only the kernel entry may return outside an inline @intent.fn")


def _lower_if(lowerer: object, node: ast.If) -> None:
    condition_expression = lowerer.lower_expression(node.test)
    known, value = compile_time_value(condition_expression)
    if known:
        lowerer.lower_statements(node.body if bool(value) else node.orelse)
        return
    condition = lowerer.materialize(
        condition_expression,
        node.test,
        ScalarType(intent_bool),
    )
    snapshot = dict(lowerer.environment)
    loop_snapshot = _copy_loop_stack(lowerer.loop_stack)
    then_region, then_environment, then_terminated, then_loops = _lower_branch(
        lowerer, node.body, snapshot, loop_snapshot, node
    )
    else_region, else_environment, else_terminated, else_loops = _lower_branch(
        lowerer, node.orelse, snapshot, loop_snapshot, node
    )
    assigned = _assigned_names(node.body) | _assigned_names(node.orelse)
    normal_environments = [
        environment
        for environment, terminated in (
            (then_environment, then_terminated),
            (else_environment, else_terminated),
        )
        if not terminated
    ]
    if not normal_environments:
        lowerer.error(node, "runtime if with both branches terminating is not representable")
    normal_loop_stacks = [
        loops
        for loops, terminated in (
            (then_loops, then_terminated),
            (else_loops, else_terminated),
        )
        if not terminated
    ]
    if lowerer.loop_stack and lowerer.loop_stack[-1].stream is not None:
        pending_states = [stack[-1].pending_stream_state for stack in normal_loop_stacks]
        if any(state != pending_states[0] for state in pending_states[1:]):
            lowerer.error(
                node,
                "runtime if cannot merge different pending stream.yield_ states; assign state and yield after if",
            )
        lowerer.loop_stack[-1].pending_stream_state = pending_states[0]

    merge_names: list[str] = []
    merged_static: dict[str, Expression] = {}
    merge_types: dict[str, object] = {}
    for name in sorted(assigned):
        if name not in snapshot and any(
            name not in environment for environment in normal_environments
        ):
            continue
        branch_values: list[Expression] = []
        for environment in normal_environments:
            if name not in environment:
                lowerer.error(node, f"runtime branch leaves {name!r} undefined")
            branch_values.append(environment[name])
        if all(_same_expression(value, branch_values[0]) for value in branch_values[1:]):
            merged_static[name] = branch_values[0]
            continue
        if all(compile_time_value(value)[0] for value in branch_values):
            static_values = [compile_time_value(value)[1] for value in branch_values]
            if all(value == static_values[0] for value in static_values[1:]):
                merged_static[name] = branch_values[0]
                continue
            if all(isinstance(value, (bool, int, float)) for value in static_values):
                if any(isinstance(value, bool) for value in static_values) and not all(
                    isinstance(value, bool) for value in static_values
                ):
                    lowerer.error(node, "runtime branch cannot mix bool and numeric literals")
                if all(isinstance(value, bool) for value in static_values):
                    merge_types[name] = ScalarType(intent_bool)
                elif any(isinstance(value, float) for value in static_values):
                    merge_types[name] = ScalarType(f64)
                else:
                    merge_types[name] = ScalarType(i64)
                merge_names.append(name)
                continue
            lowerer.error(node, f"runtime branch cannot select differing compile-time metadata {name!r}")
        merge_names.append(name)
        for branch_value in branch_values:
            branch_type = _expression_type(branch_value)
            if branch_type is not None:
                merge_types[name] = branch_type
                break

    branch_yields: list[tuple[MlirValue, ...] | None] = []
    for region, environment, terminated in (
        (then_region, then_environment, then_terminated),
        (else_region, else_environment, else_terminated),
    ):
        if terminated:
            branch_yields.append(None)
            continue
        saved_block = lowerer.current_block
        lowerer.current_block = region.blocks[0]
        values: list[MlirValue] = []
        for name in merge_names:
            expression = environment[name]
            expected = merge_types.get(name)
            value_result = lowerer.materialize(expression, node, expected)
            values.append(value_result)
        lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=tuple(values))
        lowerer.current_block = saved_block
        branch_yields.append(tuple(values))

    result_types: tuple[object, ...] = ()
    for yielded in branch_yields:
        if yielded is not None:
            result_types = tuple(value.type for value in yielded)
            break
    for yielded in branch_yields:
        if yielded is None:
            continue
        if len(yielded) != len(result_types) or any(
            not lowerer.types_compatible_for_literal(value.type, expected)
            for value, expected in zip(yielded, result_types)
        ):
            lowerer.error(node, "runtime if branch result schemas differ")
    operation = lowerer.emit(
        OperationKind.IF,
        lowerer.location(node),
        operands=(condition,),
        result_types=result_types,
        regions=(then_region, else_region),
        result_names=tuple(merge_names),
    )
    lowerer.environment = snapshot
    lowerer.environment.update(merged_static)
    lowerer.environment.update(zip(merge_names, operation.results))


def _lower_branch(
    lowerer: object,
    statements: list[ast.stmt],
    environment: dict[str, Expression],
    loop_stack: list[LoopContext],
    node: ast.AST,
) -> tuple[object, dict[str, Expression], bool, list[LoopContext]]:
    region = lowerer.compiler.builder.region(lowerer.location(node))
    saved_block = lowerer.current_block
    saved_environment = lowerer.environment
    saved_loop_stack = lowerer.loop_stack
    lowerer.current_block = region.blocks[0]
    lowerer.environment = dict(environment)
    lowerer.loop_stack = _copy_loop_stack(loop_stack)
    lowerer.lower_statements(statements)
    result_environment = dict(lowerer.environment)
    result_loop_stack = _copy_loop_stack(lowerer.loop_stack)
    terminated = lowerer.is_terminated(lowerer.current_block)
    lowerer.current_block = saved_block
    lowerer.environment = saved_environment
    lowerer.loop_stack = saved_loop_stack
    return region, result_environment, terminated, result_loop_stack


def _lower_for(lowerer: object, node: ast.For) -> None:
    if node.orelse:
        lowerer.error(node, "for-else is not part of Intent control flow")
    iteration = _lower_iteration_expression(lowerer, node.iter)
    if isinstance(iteration, StreamSpec):
        lowerer.error(node, "state_stream iteration must be enclosed by 'with stream'")
    if isinstance(iteration, IterationSpec):
        opcode = iteration.opcode
        source = iteration.source
    elif isinstance(iteration, MlirValue) and isinstance(
        iteration.type, (DomainType, RegionType, PartitionType)
    ):
        opcode = OperationKind.FOR
        source = iteration
    else:
        lowerer.error(node, "for iterator must be a domain, partition, or I.parallel")
    has_break, has_continue = (
        _loop_exit_kinds(node.body)
        if opcode is OperationKind.FOR
        else (False, False)
    )
    live_name = _fresh_loop_control_name(lowerer, node, "live") if has_break else None
    active_name = (
        _fresh_loop_control_name(lowerer, node, "active")
        if has_break or has_continue
        else None
    )
    if live_name is not None:
        lowerer.environment[live_name] = Literal(True)
    body = (
        _normalize_loop_exit_body(node.body, active_name, live_name)
        if active_name is not None
        else node.body
    )
    if live_name is not None:
        guarded = ast.If(
            test=_control_name(live_name, ast.Load(), node),
            body=body,
            orelse=[],
        )
        ast.copy_location(guarded, node)
        body = [guarded]

    snapshot = dict(lowerer.environment)
    target_names = _target_names(node.target)
    assigned_existing = (
        _assigned_names(body) & set(snapshot)
    ) - target_names
    if opcode is OperationKind.PARALLEL and assigned_existing:
        lowerer.error(
            node,
            f"parallel body cannot carry outer SSA value {sorted(assigned_existing)[0]!r}",
        )
    carried_names = tuple(sorted(assigned_existing)) if opcode is OperationKind.FOR else ()
    initial_values = tuple(
        lowerer.materialize(snapshot[name], node) for name in carried_names
    )
    iteration_types = _iteration_argument_types(lowerer, source)
    region = lowerer.compiler.builder.region(
        lowerer.location(node),
        (*iteration_types, *(value.type for value in initial_values)),
    )
    block = region.blocks[0]
    iteration_arguments = tuple(block.arguments[: len(iteration_types)])
    state_arguments = tuple(block.arguments[len(iteration_types) :])
    saved_block = lowerer.current_block
    lowerer.current_block = block
    lowerer.environment = dict(snapshot)
    _assign_iteration_target(lowerer, node.target, iteration_arguments)
    for name, value in zip(carried_names, state_arguments):
        lowerer.environment[name] = value
    lowerer.loop_stack.append(LoopContext(opcode, carried_names))
    lowerer.lower_statements(body)
    if not lowerer.is_terminated(block):
        yielded = tuple(lowerer.materialize(lowerer.environment[name], node) for name in carried_names)
        lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=yielded)
    lowerer.loop_stack.pop()
    lowerer.current_block = saved_block
    lowerer.environment = snapshot
    operation = lowerer.emit(
        opcode,
        lowerer.location(node),
        operands=(source, *initial_values),
        result_types=tuple(value.type for value in initial_values),
        regions=(region,),
        result_names=carried_names,
    )
    for name, value in zip(carried_names, operation.results):
        lowerer.environment[name] = value
    if live_name is not None:
        lowerer.environment.pop(live_name)


def _lower_iteration_expression(lowerer: object, node: ast.AST) -> Expression:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        if node.keywords or len(node.args) not in (1, 2, 3):
            lowerer.error(node, "range accepts stop or start/stop[/step]")
        if len(node.args) == 1:
            zero = ast.Constant(value=0)
            ast.copy_location(zero, node)
            arguments = [zero, node.args[0]]
        else:
            arguments = list(node.args)
        call = ast.Call(
            func=ast.Attribute(value=ast.Name(id="I", ctx=ast.Load()), attr="domain", ctx=ast.Load()),
            args=arguments,
            keywords=[],
        )
        ast.copy_location(call, node)
        from ..intrinsics.control import _domain

        return _domain(lowerer, call)
    return lowerer.lower_expression(node)


def _lower_while(lowerer: object, node: ast.While) -> None:
    if node.orelse:
        lowerer.error(node, "while-else is not part of Intent control flow")
    has_break, has_continue = _loop_exit_kinds(node.body)
    live_name = _fresh_loop_control_name(lowerer, node, "live") if has_break else None
    active_name = (
        _fresh_loop_control_name(lowerer, node, "active")
        if has_break or has_continue
        else None
    )
    if live_name is not None:
        lowerer.environment[live_name] = Literal(True)
    body = (
        _normalize_loop_exit_body(node.body, active_name, live_name)
        if active_name is not None
        else node.body
    )
    condition_node: ast.expr = node.test
    if live_name is not None:
        condition_node = ast.IfExp(
            test=_control_name(live_name, ast.Load(), node.test),
            body=node.test,
            orelse=ast.Constant(value=False),
        )
        ast.copy_location(condition_node, node.test)

    snapshot = dict(lowerer.environment)
    carried_names = tuple(sorted(_assigned_names(body) & set(snapshot)))
    initial_values = tuple(lowerer.materialize(snapshot[name], node) for name in carried_names)
    state_types = tuple(value.type for value in initial_values)
    before = lowerer.compiler.builder.region(lowerer.location(node), state_types)
    after = lowerer.compiler.builder.region(lowerer.location(node), state_types)
    saved_block = lowerer.current_block

    lowerer.current_block = before.blocks[0]
    lowerer.environment = dict(snapshot)
    for name, value in zip(carried_names, before.blocks[0].arguments):
        lowerer.environment[name] = value
    condition_expression = lowerer.lower_expression(condition_node)
    condition = lowerer.materialize(condition_expression, node.test, ScalarType(intent_bool))
    lowerer.emit(
        OperationKind.CONDITION,
        lowerer.location(node.test),
        operands=(condition, *before.blocks[0].arguments),
    )

    lowerer.current_block = after.blocks[0]
    lowerer.environment = dict(snapshot)
    for name, value in zip(carried_names, after.blocks[0].arguments):
        lowerer.environment[name] = value
    lowerer.loop_stack.append(LoopContext(OperationKind.WHILE, carried_names))
    lowerer.lower_statements(body)
    if not lowerer.is_terminated(after.blocks[0]):
        yielded = tuple(lowerer.materialize(lowerer.environment[name], node) for name in carried_names)
        lowerer.emit(OperationKind.YIELD, lowerer.location(node), operands=yielded)
    lowerer.loop_stack.pop()

    lowerer.current_block = saved_block
    lowerer.environment = snapshot
    operation = lowerer.emit(
        OperationKind.WHILE,
        lowerer.location(node),
        operands=initial_values,
        result_types=state_types,
        regions=(before, after),
        result_names=carried_names,
    )
    for name, result in zip(carried_names, operation.results):
        lowerer.environment[name] = result
    if live_name is not None:
        lowerer.environment.pop(live_name)


def _lower_with(lowerer: object, node: ast.With) -> None:
    if len(node.items) != 1 or node.items[0].optional_vars is not None:
        lowerer.error(node, "state_stream with block accepts one stream and no 'as' target")
    stream_expression = lowerer.lower_expression(node.items[0].context_expr)
    if not isinstance(stream_expression, StreamSpec):
        lowerer.error(node, "with is reserved for I.state_stream handles")
    if len(node.body) != 1 or not isinstance(node.body[0], ast.For):
        lowerer.error(node, "with stream body must contain exactly one 'for ... in stream'")
    loop = node.body[0]
    iterator = lowerer.lower_expression(loop.iter)
    if iterator is not stream_expression:
        lowerer.error(loop.iter, "stream loop must iterate the same stream handle")
    if loop.orelse:
        lowerer.error(loop, "state_stream loop does not support else")
    stream = stream_expression
    if isinstance(stream.axis.type, RegionType):
        segment_type = stream.axis.type
    elif stream.axis.type.flavor is DomainFlavor.RAGGED_MEMBER:
        segment_type = RegionType(stream.axis.type.rank, "ragged_member")
    elif stream.axis.type.flavor is DomainFlavor.RAGGED_OUTER:
        segment_type = RegionType(stream.axis.type.rank, "ragged_outer")
    else:
        segment_type = RegionType(stream.axis.type.rank, "state_stream")
    region = lowerer.compiler.builder.region(
        lowerer.location(loop),
        (segment_type, *(value.type for value in stream.initial_state)),
    )
    snapshot = dict(lowerer.environment)
    saved_block = lowerer.current_block
    lowerer.current_block = region.blocks[0]
    lowerer.environment = dict(snapshot)
    arguments = tuple(region.blocks[0].arguments)
    if isinstance(loop.target, (ast.Tuple, ast.List)) and len(loop.target.elts) == 2:
        _assign_target(lowerer, loop.target.elts[0], arguments[0])
        state_expression: Expression = (
            arguments[1]
            if len(arguments) == 2
            else StaticTuple(tuple(arguments[1:]))
        )
        _assign_target(lowerer, loop.target.elts[1], state_expression)
    else:
        _assign_iteration_target(lowerer, loop.target, arguments)
    context = LoopContext(OperationKind.STATE_STREAM, (), stream=stream)
    lowerer.loop_stack.append(context)
    lowerer.lower_statements(loop.body)
    if not lowerer.is_terminated(region.blocks[0]):
        if context.pending_stream_state is None:
            lowerer.error(loop, "every state_stream path must call stream.yield_(...)")
        lowerer.emit(
            OperationKind.YIELD,
            lowerer.location(loop),
            operands=context.pending_stream_state,
        )
    lowerer.loop_stack.pop()
    lowerer.current_block = saved_block
    lowerer.environment = snapshot
    operands = [stream.axis, *stream.initial_state]
    attributes: dict[str, object] = {"state_count": len(stream.initial_state)}
    if isinstance(stream.extent, MlirValue):
        attributes["extent_operand_index"] = len(operands)
        operands.append(stream.extent)
    else:
        attributes["extent"] = stream.extent
    if stream.stop is not None:
        attributes["stop_operand_index"] = len(operands)
        operands.append(stream.stop)
    operation = lowerer.emit(
        OperationKind.STATE_STREAM,
        lowerer.location(node),
        operands=tuple(operands),
        result_types=tuple(value.type for value in stream.initial_state),
        attributes=attributes,
        regions=(region,),
    )
    stream.results = operation.results


def _lower_assert(lowerer: object, node: ast.Assert) -> None:
    condition = lowerer.lower_expression(node.test)
    known, value = compile_time_value(condition)
    if not known:
        lowerer.error(node, "runtime assert is wrapper policy, not kernel control flow")
    if not bool(value):
        lowerer.error(node, "compile-time assertion failed")


def _iteration_argument_types(lowerer: object, source: MlirValue) -> tuple[object, ...]:
    if isinstance(source.type, PartitionType):
        if source.type.mode is PartitionMode.COUNT:
            return (LogicalIndexType("partition_part"), source.type.region_type)
        return (source.type.region_type,)
    if isinstance(source.type, DomainType):
        return tuple(
            LogicalIndexType(f"domain_axis_{axis}") for axis in range(source.type.rank)
        )
    if isinstance(source.type, RegionType):
        return tuple(
            LogicalIndexType(f"{source.type.relation}_axis_{axis}")
            for axis in range(source.type.rank)
        )
    lowerer.error(source.location, "invalid iteration source")


def _assign_iteration_target(
    lowerer: object,
    target: ast.AST,
    arguments: tuple[MlirValue, ...],
) -> None:
    expression: Expression
    if len(arguments) == 1:
        expression = arguments[0]
    else:
        expression = StaticTuple(tuple(arguments))
    _assign_target(lowerer, target, expression)


def _assigned_names(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()

    class Collector(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store) and node.id != "_":
                names.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    collector = Collector()
    for statement in statements:
        collector.visit(statement)
    return names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return set() if target.id == "_" else {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _loop_exit_kinds(statements: list[ast.stmt]) -> tuple[bool, bool]:
    class Collector(ast.NodeVisitor):
        has_break = False
        has_continue = False

        def visit_Break(self, node: ast.Break) -> None:
            self.has_break = True

        def visit_Continue(self, node: ast.Continue) -> None:
            self.has_continue = True

        def visit_For(self, node: ast.For) -> None:
            return

        def visit_While(self, node: ast.While) -> None:
            return

        def visit_With(self, node: ast.With) -> None:
            return

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    collector = Collector()
    for statement in statements:
        collector.visit(statement)
    return collector.has_break, collector.has_continue


def _fresh_loop_control_name(lowerer: object, node: ast.AST, role: str) -> str:
    line = getattr(node, "lineno", 0)
    column = getattr(node, "col_offset", 0)
    candidate = f"__intent_{role}_{line}_{column}"
    occupied = set(lowerer.environment)
    occupied.update(
        child.id for child in ast.walk(node) if isinstance(child, ast.Name)
    )
    while candidate in occupied:
        candidate += "_"
    return candidate


def _control_name(name: str, context: ast.expr_context, location: ast.AST) -> ast.Name:
    value = ast.Name(id=name, ctx=context)
    ast.copy_location(value, location)
    return value


def _control_assignment(name: str, value: bool, location: ast.AST) -> ast.Assign:
    assignment = ast.Assign(
        targets=[_control_name(name, ast.Store(), location)],
        value=ast.Constant(value=value),
    )
    ast.copy_location(assignment.value, location)
    ast.copy_location(assignment, location)
    return assignment


def _normalize_loop_exit_body(
    statements: list[ast.stmt],
    active_name: str,
    live_name: str | None,
) -> list[ast.stmt]:
    location = statements[0]
    return [
        _control_assignment(active_name, True, location),
        *_normalize_guarded_block(statements, active_name, live_name),
    ]


def _rewrite_loop_exit_statement(
    statement: ast.stmt,
    active_name: str,
    live_name: str | None,
) -> list[ast.stmt]:
    if isinstance(statement, ast.Break):
        return [
            _control_assignment(cast(str, live_name), False, statement),
            _control_assignment(active_name, False, statement),
        ]
    if isinstance(statement, ast.Continue):
        return [_control_assignment(active_name, False, statement)]
    if isinstance(statement, ast.If):
        rewritten = ast.If(
            test=statement.test,
            body=_normalize_guarded_block(statement.body, active_name, live_name),
            orelse=_normalize_guarded_block(statement.orelse, active_name, live_name),
        )
        ast.copy_location(rewritten, statement)
        return [rewritten]
    return [statement]


def _normalize_guarded_block(
    statements: list[ast.stmt],
    active_name: str,
    live_name: str | None,
) -> list[ast.stmt]:
    normalized: list[ast.stmt] = []
    for position, statement in enumerate(statements):
        has_break, has_continue = _loop_exit_kinds([statement])
        if not has_break and not has_continue:
            normalized.append(statement)
            continue
        normalized.extend(
            _rewrite_loop_exit_statement(statement, active_name, live_name)
        )
        remaining = statements[position + 1 :]
        if remaining:
            guarded = ast.If(
                test=_control_name(active_name, ast.Load(), remaining[0]),
                body=_normalize_guarded_block(remaining, active_name, live_name),
                orelse=[],
            )
            ast.copy_location(guarded, remaining[0])
            normalized.append(guarded)
        break
    return normalized


def _copy_loop_stack(stack: list[LoopContext]) -> list[LoopContext]:
    return [
        LoopContext(
            context.opcode,
            context.carried_names,
            context.stream,
            context.pending_stream_state,
        )
        for context in stack
    ]


def _same_expression(lhs: Expression, rhs: Expression) -> bool:
    if isinstance(lhs, MlirValue) or isinstance(rhs, MlirValue):
        return lhs is rhs
    if isinstance(lhs, ConstexprBinding) or isinstance(rhs, ConstexprBinding):
        return lhs is rhs
    return lhs == rhs


def _expression_type(expression: Expression) -> object | None:
    if isinstance(expression, MlirValue):
        return expression.type
    if isinstance(expression, ConstexprBinding):
        return expression.ir_value.type
    return None
