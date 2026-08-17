from __future__ import annotations

from types import SimpleNamespace


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
        "scan": (32, 64, 128, 256, 512, 1024),
        "stream_contract": (32, 64, 128),
        "stream_scaled": (1, 2, 4, 8),
        "query": (1, 2, 64, 128),
        "ragged_member": (64, 128),
        "lane_pack": (64, 128, 256),
        "feature": (64, 128),
        "reduction": (16, 32, 64),
        "program_m": (16, 32, 64, 128),
        "program_n": (16, 32, 64, 128),
        "group_m": (4, 8),
        "gather_spelling": (0, 1),
    }.get(role)
    if candidates is not None:
        return candidates
    for base in (
        "stream_contract",
        "scan",
        "stream",
        "query",
        "ragged_member",
        "lane_pack",
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


def _completion_candidates(role: str) -> tuple[int, ...]:
    candidates = _role_candidates(role)
    if role == "query" or role.startswith("query_"):
        return tuple(value for value in candidates if value not in (1, 2))
    return candidates


def autotune_configurations(parameter_map: dict[str, str]) -> tuple[SimpleNamespace, ...]:
    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    joint_program_profiles = (
        tuple(
            (
                {"program_m": program_m, "program_n": program_n},
                1,
                occupancy,
            )
            for program_m, program_n in (
                (16, 16),
                (32, 32),
                (64, 64),
                (64, 128),
                (128, 64),
                (128, 128),
            )
            for occupancy in (1, 2, 4)
        )
        if roles == {"program_m", "program_n"}
        else ()
    )
    profiles = (
        (
            ({"scan": 64}, 1, 4),
            ({"scan": 128}, 1, 2),
            ({"scan": 256}, 2, 1),
        ),
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
            ({"stream_scaled": 1}, 1, 4),
            ({"stream_scaled": 2}, 1, 2),
            ({"stream_scaled": 4}, 1, 1),
            ({"stream_scaled": 8}, 2, 1),
        ),
        tuple(
            ({"query": query, stream_role: stream}, num_ctas, occupancy)
            for stream_role in ("stream", "stream_contract")
            for query, stream, num_ctas, occupancy in (
                (1, 32, 1, 4),
                (2, 32, 1, 4),
                (128, 128, 1, 2),
                (128, 128, 2, 2),
                (64, 64, 1, 4),
                (64, 32, 1, 2),
            )
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
            for reduction in (16, 32, 64)
            for occupancy in (1, 2, 4)
        ),
        joint_program_profiles,
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
            role: profile.get(
                role,
                _completion_candidates(role)[
                    index % len(_completion_candidates(role))
                ],
            )
            for role in roles
            if role != "gather_spelling"
        }
        key = (tuple(sorted(values.items())), num_ctas, occupancy)
        if key not in seen:
            seen.add(key)
            choices.append((values, num_ctas, occupancy))
    if "gather_spelling" in roles:
        choices = [
            (
                {**values, "gather_spelling": spelling},
                num_ctas,
                occupancy,
            )
            for values, num_ctas, occupancy in choices
            for spelling in _role_candidates("gather_spelling")
        ]
    return tuple(
        _configuration(
            parameter_map,
            values,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )
        for values, num_ctas, occupancy in choices
    )


def autotune_timeout(parameter_map: dict[str, str]) -> int:
    roles = frozenset(parameter_map.values())
    return (
        10
        if "query" in roles and {"stream", "stream_contract"}.intersection(roles)
        else 5
    )


def tune_row_occupancy(
    stream: object,
    grid: tuple[int, int, int],
    kernel: object,
    args_fn: object,
) -> object:
    import cuda.tile as ct
    from cuda.tile.tune import exhaustive_search

    configurations = tuple(
        SimpleNamespace(occupancy=occupancy) for occupancy in (0, 1, 2, 4)
    )
    with ct.compiler_timeout(5):
        result = exhaustive_search(
            configurations,
            stream,
            lambda _: grid,
            kernel,
            args_fn,
            hints_fn=lambda config: (
                {} if config.occupancy == 0 else {"occupancy": config.occupancy}
            ),
            quiet=True,
            single_run_timeout_sec=5,
        )
    occupancy = result.best.config.occupancy
    return kernel if occupancy == 0 else kernel.replace_hints(occupancy=occupancy)


def tune_persistent_row(
    stream: object,
    n_rows: int,
    device: object,
    kernel: object,
    args_fn: object,
) -> SimpleNamespace:
    import cuda.tile as ct
    import torch
    from cuda.tile.tune import exhaustive_search

    compute_units = torch.cuda.get_device_properties(device).multi_processor_count
    configurations = tuple(
        SimpleNamespace(occupancy=occupancy, grid_occupancy=max(1, occupancy))
        for occupancy in (0, 1, 2, 4)
    )
    grid_fn = lambda config: (
        min(compute_units * config.grid_occupancy, n_rows),
        1,
        1,
    )
    with ct.compiler_timeout(5):
        result = exhaustive_search(
            configurations,
            stream,
            grid_fn,
            kernel,
            args_fn,
            hints_fn=lambda config: (
                {} if config.occupancy == 0 else {"occupancy": config.occupancy}
            ),
            quiet=True,
            single_run_timeout_sec=5,
        )
    occupancy = result.best.config.occupancy
    return SimpleNamespace(
        kernel=(kernel if occupancy == 0 else kernel.replace_hints(occupancy=occupancy)),
        grid=grid_fn(result.best.config),
    )
