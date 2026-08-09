from __future__ import annotations

from types import SimpleNamespace


def _target_parameters(
    parameter_map: dict[str, str], values: dict[str, int]
) -> dict[str, int]:
    return {target: values[role] for target, role in parameter_map.items()}


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 256, 512, 1024),
        "query": (64, 128),
        "ragged_member": (32, 64, 128),
        "feature": (64, 128, 256),
        "reduction": (32, 64, 128),
        "program_m": (32, 64, 128, 256),
        "program_n": (32, 64, 128, 256),
        "group_m": (4, 8),
    }.get(role)
    if candidates is not None:
        return candidates
    for base in ("stream", "query", "ragged_member", "feature", "reduction"):
        if role.startswith(f"{base}_"):
            return _role_candidates(base)
    if role.startswith("program_"):
        return (32, 64, 128)
    if role.startswith("group_"):
        return (2, 4, 8)
    raise NotImplementedError(f"unsupported Triton tuner role: {role}")


def autotune_configurations(parameter_map: dict[str, str]) -> list[object]:
    import triton

    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    profiles = (
        (
            ({"stream": 256}, 4, 4),
            ({"stream": 512}, 3, 8),
            ({"stream": 1024}, 2, 8),
        ),
        (
            ({"query": 64, "stream": 32}, 3, 4),
            ({"query": 64, "stream": 64}, 3, 4),
            ({"query": 128, "stream": 32}, 3, 4),
            ({"query": 128, "stream": 64}, 3, 4),
            ({"query": 128, "stream": 64}, 2, 8),
            ({"query": 128, "stream": 128}, 2, 8),
        ),
        (
            ({"ragged_member": 32, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 64, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 64, "feature": 128, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 64, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 128, "reduction": 32}, 4, 4),
            ({"ragged_member": 128, "feature": 128, "reduction": 64}, 3, 8),
        ),
        tuple(
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
            for m, n, k, group, stages, warps in (
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
    for index, (profile, stages, warps) in enumerate(selected):
        values = {
            role: profile.get(role, _role_candidates(role)[index % len(_role_candidates(role))])
            for role in roles
        }
        key = (tuple(sorted(values.items())), stages, warps)
        if key not in seen:
            seen.add(key)
            choices.append((values, stages, warps))
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
