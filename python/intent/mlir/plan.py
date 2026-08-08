from __future__ import annotations

from intent.realizer import PhysicalPlan

from .types import quote


def emit_physical_plan(plan: PhysicalPlan, indent: int = 1) -> list[str]:
    prefix = "  " * indent
    body = "  " * (indent + 2)
    lines = [f'{prefix}"intent_plan.plan"() (', f"{prefix}  {{", f"{prefix}    ^bb0:"]

    for binding in plan.extents:
        lines.append(
            body
            + '"intent_plan.extent"() {'
            + f"axis = {binding.axis} : i64, "
            + f"logical = {quote(binding.logical_extent)}, "
            + f"node = {binding.node_id} : i64, "
            + f"tile = {quote(binding.tile.value)}"
            + "} : () -> ()"
        )
    for binding in plan.ownership:
        lines.append(
            body
            + '"intent_plan.ownership"() {'
            + f"loop_node = {binding.loop_node_id} : i64, "
            + f"mapping = {quote(binding.mapping.value)}, "
            + f"traversal = {quote(binding.traversal.value)}, "
            + f"worker = {quote(binding.worker.value)}, "
            + f"worker_axis = {binding.worker_axis} : i64"
            + "} : () -> ()"
        )
    for binding in plan.storage:
        lines.append(
            body
            + '"intent_plan.storage"() {'
            + f"access = {quote(binding.access.value)}, "
            + f"space = {quote(binding.space.value)}, "
            + f"value = {binding.value_id} : i64"
            + "} : () -> ()"
        )
    for binding in plan.layouts:
        order = ", ".join(str(axis) for axis in binding.order)
        lines.append(
            body
            + '"intent_plan.layout"() {'
            + f"kind = {quote(binding.kind.value)}, "
            + f"order = array<i64: {order}>, "
            + f"value = {binding.value_id} : i64"
            + "} : () -> ()"
        )
    for binding in plan.primitives:
        lines.append(
            body
            + '"intent_plan.primitive"() {'
            + f"axis = {binding.axis} : i64, "
            + f"identity = {quote(binding.identity)}, "
            + f"kind = {quote(binding.kind.value)}, "
            + f"node = {binding.node_id} : i64, "
            + f"operator_name = {quote(binding.operator)}"
            + "} : () -> ()"
        )
    for binding in plan.boundaries:
        lines.append(
            body
            + '"intent_plan.boundary"() {'
            + f"load_fill = {quote(binding.load_fill)}, "
            + f"logical = {quote(binding.logical_extent)}, "
            + f"node = {binding.node_id} : i64, "
            + f"predicate = {quote(binding.predicate)}, "
            + f"tail = {quote(binding.tail.value)}"
            + "} : () -> ()"
        )
    pipeline = plan.pipeline
    lines.append(
        body
        + '"intent_plan.pipeline"() {'
        + f"async_copy = {'true' if pipeline.async_copy else 'false'}, "
        + f"high_stages = {pipeline.high_stages} : i64, "
        + f"low_stages = {pipeline.low_stages} : i64, "
        + f"prefetch = {'true' if pipeline.prefetch else 'false'}, "
        + f"smem_threshold = {pipeline.smem_threshold} : i64"
        + "} : () -> ()"
    )
    launch = plan.launch
    lines.append(
        body
        + '"intent_plan.launch"() {'
        + f"grid_policy = {quote(launch.grid_policy.value)}, "
        + f"loop_node = {launch.loop_node_id} : i64, "
        + f"num_warps = {launch.num_warps} : i64"
        + "} : () -> ()"
    )
    lines.append(body + '"intent_plan.yield"() : () -> ()')
    lines.extend(
        [
            f"{prefix}  }}",
            f"{prefix}) {{architecture = {quote(plan.target.architecture)}, "
            + f"backend = {quote(plan.target.backend.value)}, "
            + f"device = {plan.target.device} : i64, entry = @{plan.entry_name}, "
            + f"warp_size = {plan.target.warp_size} : i64}} : () -> ()",
        ]
    )
    return lines
