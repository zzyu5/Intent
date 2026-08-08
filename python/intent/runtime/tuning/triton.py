from __future__ import annotations

from types import SimpleNamespace


def _target_parameters(
    parameter_map: dict[str, str], values: dict[str, int]
) -> dict[str, int]:
    return {target: values[role] for target, role in parameter_map.items()}


def autotune_configurations(parameter_map: dict[str, str]) -> list[object]:
    import triton

    roles = frozenset(parameter_map.values())
    if roles == {"stream"}:
        choices = (
            ({"stream": 256}, 4, 4),
            ({"stream": 512}, 3, 8),
            ({"stream": 1024}, 2, 8),
        )
    elif roles == {"query", "stream"}:
        choices = (
            ({"query": 64, "stream": 32}, 3, 4),
            ({"query": 64, "stream": 64}, 3, 4),
            ({"query": 128, "stream": 32}, 3, 4),
            ({"query": 128, "stream": 64}, 3, 4),
            ({"query": 128, "stream": 64}, 2, 8),
            ({"query": 128, "stream": 128}, 2, 8),
        )
    elif roles == {"ragged_member", "feature", "reduction"}:
        choices = (
            ({"ragged_member": 32, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 64, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 64, "feature": 128, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 128, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 128, "reduction": 64}, 3, 8),
        )
    elif roles == {"program_m", "program_n", "reduction", "group_m"}:
        raw = (
            (128, 256, 64, 8, 3, 8),
            (64, 256, 32, 8, 4, 4),
            (128, 128, 32, 8, 4, 4),
            (128, 64, 32, 8, 4, 4),
            (64, 128, 32, 8, 4, 4),
            (128, 32, 32, 8, 4, 4),
            (64, 32, 32, 8, 5, 2),
            (32, 64, 32, 8, 5, 2),
            (128, 256, 128, 8, 3, 8),
            (256, 128, 128, 8, 3, 8),
            (256, 64, 128, 8, 4, 4),
            (64, 256, 128, 8, 4, 4),
            (128, 128, 128, 8, 4, 4),
            (128, 64, 64, 8, 4, 4),
            (64, 128, 64, 8, 4, 4),
            (128, 32, 64, 8, 4, 4),
        )
        choices = tuple(
            (
                {
                    "program_m": m,
                    "program_n": n,
                    "reduction": k,
                    "group_m": group,
                },
                stages,
                warps,
            )
            for m, n, k, group, stages, warps in raw
        )
    else:
        raise NotImplementedError(f"unsupported Triton tuner roles: {sorted(roles)}")
    return [
        triton.Config(
            _target_parameters(parameter_map, values),
            num_stages=stages,
            num_warps=warps,
        )
        for values, stages, warps in choices
    ]


def row_configuration(n_columns: int, properties: dict[str, int]) -> SimpleNamespace:
    import triton

    return SimpleNamespace(
        tile_size=triton.next_power_of_2(n_columns),
        num_stages=4 if properties["max_shared_mem"] > 200000 else 2,
        num_warps=8,
    )


def row_program_count(
    n_rows: int,
    compiled_kernel: object,
    properties: dict[str, int],
    warp_size: int,
    configuration: SimpleNamespace,
) -> int:
    occupancy = properties["max_num_regs"] // (
        compiled_kernel.n_regs * warp_size * configuration.num_warps
    )
    occupancy = min(occupancy, properties["max_shared_mem"] // compiled_kernel.metadata.shared)
    return min(properties["multiprocessor_count"] * occupancy, n_rows)
