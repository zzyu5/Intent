from __future__ import annotations

from types import SimpleNamespace

DEFAULT_NUM_STAGES = 1
DEFAULT_THREADS = 128


def _configuration(
    parameter_map: dict[str, str],
    values: dict[str, int],
    *,
    num_stages: int,
    threads: int,
) -> dict[str, int]:
    target = {name: values[role] for name, role in parameter_map.items()}
    return {**target, "num_stages": num_stages, "threads": threads}


def autotune_configurations(parameter_map: dict[str, str]) -> list[dict[str, int]]:
    roles = frozenset(parameter_map.values())
    if roles == {"stream"}:
        choices = (
            ({"stream": 256}, 1, 128),
            ({"stream": 512}, 1, 128),
            ({"stream": 1024}, 2, 256),
        )
    elif roles == {"query", "stream"}:
        choices = (
            ({"query": 64, "stream": 64}, 1, 128),
            ({"query": 128, "stream": 64}, 1, 128),
            ({"query": 128, "stream": 128}, 1, 128),
        )
    elif roles == {"program_m", "program_n", "reduction", "group_m"}:
        choices = (
            ({"program_m": 128, "program_n": 64, "reduction": 64, "group_m": 8}, 2, 128),
            ({"program_m": 128, "program_n": 128, "reduction": 32, "group_m": 8}, 3, 256),
            ({"program_m": 64, "program_n": 128, "reduction": 64, "group_m": 8}, 2, 128),
        )
    elif roles == {"ragged_member", "feature", "reduction"}:
        choices = (
            ({"ragged_member": 128, "feature": 64, "reduction": 64}, 2, 128),
            ({"ragged_member": 128, "feature": 128, "reduction": 32}, 3, 256),
            ({"ragged_member": 64, "feature": 128, "reduction": 64}, 2, 128),
        )
    else:
        raise NotImplementedError(f"unsupported TileLang tuner roles: {sorted(roles)}")
    return [
        _configuration(
            parameter_map,
            values,
            num_stages=num_stages,
            threads=threads,
        )
        for values, num_stages, threads in choices
    ]


def row_configuration(n_columns: int) -> SimpleNamespace:
    return SimpleNamespace(
        tile_size=1 << (n_columns - 1).bit_length(),
        num_stages=DEFAULT_NUM_STAGES,
        threads=DEFAULT_THREADS,
    )
