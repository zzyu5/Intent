from __future__ import annotations


def _target_parameters(
    parameter_map: dict[str, str], values: dict[str, int]
) -> dict[str, int]:
    return {target: values[role] for target, role in parameter_map.items()}


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
        "scan": (32, 64, 128, 256, 512, 1024),
        "stream_contract": (32, 64, 128),
        "query": (1, 2, 16, 32, 64, 128),
        "ragged_member": (32, 64, 128),
        "lane_pack": (64, 128, 256, 512),
        "feature": (64, 128, 256),
        "reduction": (32, 64, 128),
        "program_m": (32, 64, 128, 256),
        "program_n": (32, 64, 128, 256),
        "group_m": (4, 8),
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
    raise NotImplementedError(f"unsupported Triton tuner role: {role}")


def _completion_candidates(role: str) -> tuple[int, ...]:
    candidates = _role_candidates(role)
    if role == "query" or role.startswith("query_"):
        return tuple(value for value in candidates if value not in (1, 2))
    return candidates


def autotune_configurations(parameter_map: dict[str, str]) -> list[object]:
    import triton

    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    profiles = (
        (
            ({"scan": 64}, 1, 4),
            ({"scan": 128}, 2, 4),
            ({"scan": 256}, 2, 8),
        ),
        (
            ({"stream": 256}, 4, 4),
            ({"stream": 512}, 3, 8),
            ({"stream": 1024}, 2, 8),
            ({"stream": 2048}, 2, 8),
            ({"stream": 4096}, 2, 8),
            ({"stream": 8192}, 2, 16),
            ({"stream": 16384}, 1, 32),
            ({"stream": 32768}, 1, 32),
        ),
        (
            ({"stream_contract": 32}, 4, 4),
            ({"stream_contract": 64}, 3, 4),
        ),
        tuple(
            ({"query": query, stream_role: stream}, stages, warps)
            for stream_role in ("stream", "stream_contract")
            for query, stream, stages, warps in (
                (1, 32, 1, 4),
                (2, 32, 1, 4),
                (32, 32, 1, 4),
                (64, 32, 1, 4),
                (64, 32, 3, 4),
                (64, 64, 3, 4),
                (128, 32, 3, 4),
                (128, 64, 3, 4),
                (128, 64, 2, 8),
                (128, 128, 2, 8),
            )
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
            role: profile.get(
                role,
                _completion_candidates(role)[
                    index % len(_completion_candidates(role))
                ],
            )
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


def row_autotune_configurations(*, persistent: bool = False) -> list[object]:
    import triton

    if persistent:
        profiles = (
            (4, 1, 4),
            (4, 2, 2),
            (4, 3, 2),
            (8, 2, 4),
            (8, 2, 2),
            (8, 3, 1),
            (8, 4, 1),
            (16, 2, 1),
            (16, 3, 1),
            (32, 2, 1),
        )
        return [
            triton.Config(
                {"ROW_OCCUPANCY": occupancy, "PIPELINE_STAGES": stages},
                num_stages=stages,
                num_warps=warps,
            )
            for warps, stages, occupancy in profiles
        ]

    profiles = (
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (8, 2),
        (8, 3),
        (8, 4),
        (16, 2),
        (16, 3),
        (32, 2),
    )
    return [
        triton.Config({}, num_stages=stages, num_warps=warps)
        for warps, stages in profiles
    ]
