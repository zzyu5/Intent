from __future__ import annotations

from types import SimpleNamespace

ROW_OCCUPANCY = 4


def _configuration(
    parameter_map: dict[str, str],
    values: dict[str, int],
    *,
    num_ctas: int,
    occupancy: int,
) -> SimpleNamespace:
    target = {name: values[role] for name, role in parameter_map.items()}
    return SimpleNamespace(**target, num_ctas=num_ctas, occupancy=occupancy)


def autotune_configurations(parameter_map: dict[str, str]) -> tuple[SimpleNamespace, ...]:
    roles = frozenset(parameter_map.values())
    if roles == {"stream"}:
        choices = (
            ({"stream": 512}, 1, 4),
            ({"stream": 1024}, 1, 2),
            ({"stream": 2048}, 2, 1),
        )
    elif roles == {"query", "stream"}:
        choices = (
            ({"query": 128, "stream": 128}, 1, 2),
            ({"query": 128, "stream": 128}, 2, 2),
            ({"query": 64, "stream": 64}, 1, 4),
            ({"query": 64, "stream": 32}, 1, 2),
        )
    elif roles == {"program_m", "program_n", "reduction", "group_m"}:
        choices = (
            ({"program_m": 128, "program_n": 64, "reduction": 64, "group_m": 8}, 1, 1),
            ({"program_m": 128, "program_n": 64, "reduction": 32, "group_m": 8}, 1, 2),
        )
    elif roles == {"ragged_member", "feature", "reduction"}:
        choices = (
            ({"ragged_member": 128, "feature": 64, "reduction": 64}, 1, 1),
            ({"ragged_member": 128, "feature": 64, "reduction": 32}, 1, 2),
        )
    else:
        raise NotImplementedError(f"unsupported cuTile tuner roles: {sorted(roles)}")
    return tuple(
        _configuration(
            parameter_map,
            values,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )
        for values, num_ctas, occupancy in choices
    )


def row_configuration(n_columns: int) -> SimpleNamespace:
    return SimpleNamespace(
        tile_size=1 << (n_columns - 1).bit_length(), occupancy=ROW_OCCUPANCY
    )


def row_program_count(n_rows: int, device: object, occupancy: int) -> int:
    import torch

    compute_units = torch.cuda.get_device_properties(device).multi_processor_count
    return min(compute_units * occupancy, n_rows)


def autotune_timeout(parameter_map: dict[str, str]) -> int:
    return 10 if frozenset(parameter_map.values()) == {"query", "stream"} else 5
