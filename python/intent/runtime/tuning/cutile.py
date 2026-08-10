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


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 512, 1024, 2048),
        "stream_contract": (32, 64),
        "query": (64, 128),
        "ragged_member": (64, 128),
        "feature": (64, 128),
        "reduction": (32, 64),
        "program_m": (64, 128),
        "program_n": (64, 128),
        "group_m": (4, 8),
    }.get(role)
    if candidates is not None:
        return candidates
    for base in (
        "stream_contract",
        "stream",
        "query",
        "ragged_member",
        "feature",
        "reduction",
    ):
        if role.startswith(f"{base}_"):
            return _role_candidates(base)
    if role.startswith("program_"):
        return (32, 64, 128)
    if role.startswith("group_"):
        return (2, 4, 8)
    raise NotImplementedError(f"unsupported cuTile tuner role: {role}")


def autotune_configurations(parameter_map: dict[str, str]) -> tuple[SimpleNamespace, ...]:
    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    profiles = (
        (
            ({"stream": 512}, 1, 4),
            ({"stream": 1024}, 1, 2),
            ({"stream": 2048}, 2, 1),
        ),
        (
            ({"stream_contract": 32}, 1, 4),
            ({"stream_contract": 64}, 1, 2),
        ),
        (
            ({"query": 128, "stream": 128}, 1, 2),
            ({"query": 128, "stream": 128}, 2, 2),
            ({"query": 64, "stream": 64}, 1, 4),
            ({"query": 64, "stream": 32}, 1, 2),
        ),
        tuple(
            (
                {
                    "program_m": program_m,
                    "program_n": program_n,
                    "reduction": reduction,
                    "group_m": 8,
                },
                1,
                occupancy,
            )
            for program_m in (64, 128)
            for program_n in (64, 128)
            for reduction in (32, 64)
            for occupancy in (1, 2, 4)
        ),
        (
            ({"ragged_member": 128, "feature": 64, "reduction": 64}, 1, 1),
            ({"ragged_member": 128, "feature": 64, "reduction": 32}, 1, 2),
        ),
    )
    score = max(
        (len(roles.intersection(values)), -len(set(values).difference(roles)))
        for family in profiles
        for values, _, _ in family
    )
    selected = [
        profile
        for family in profiles
        for profile in family
        if (
            len(roles.intersection(profile[0])),
            -len(set(profile[0]).difference(roles)),
        )
        == score
    ]
    choices = []
    seen = set()
    for index, (profile, num_ctas, occupancy) in enumerate(selected):
        values = {
            role: profile.get(role, _role_candidates(role)[index % len(_role_candidates(role))])
            for role in roles
        }
        key = (tuple(sorted(values.items())), num_ctas, occupancy)
        if key not in seen:
            seen.add(key)
            choices.append((values, num_ctas, occupancy))
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
    roles = frozenset(parameter_map.values())
    return 10 if {"query", "stream"}.issubset(roles) else 5
