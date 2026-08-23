from __future__ import annotations

from itertools import product


def _target_parameters(
    parameter_map: dict[str, str], values: dict[str, int]
) -> dict[str, int]:
    return {target: values[role] for target, role in parameter_map.items()}


def _power_of_two_ceiling(value: int) -> int:
    if value < 1:
        raise ValueError("Triton parameter extents must be positive")
    return 1 << (value - 1).bit_length()


def runtime_extent_pruning(
    parameter_extents: dict[str, tuple[str, ...]],
    descriptor_views: tuple[str, ...] = (),
) -> dict[str, object]:
    bindings = {
        parameter: tuple(extents)
        for parameter, extents in parameter_extents.items()
    }

    def early_config_prune(configs, named_args, **kwargs):
        arguments = {**named_args, **kwargs}
        descriptors_legal = all(
            view in arguments
            and hasattr(arguments[view], "is_contiguous")
            and arguments[view].is_contiguous()
            and arguments[view].data_ptr() % 16 == 0
            and arguments[view].ndim >= 2
            and arguments[view].shape[-1] * arguments[view].element_size() % 16 == 0
            for view in descriptor_views
        )
        limits = {}
        for parameter, extents in bindings.items():
            if not extents:
                raise ValueError(
                    f"Triton runtime extent constraint for {parameter!r} is empty"
                )
            missing = tuple(extent for extent in extents if extent not in arguments)
            if missing:
                raise ValueError(
                    f"Triton runtime extent constraint for {parameter!r} "
                    f"references unknown kernel arguments {missing!r}"
                )
            minimum_candidate = min(
                config.kwargs[parameter] for config in configs
            )
            limits[parameter] = max(
                minimum_candidate,
                _power_of_two_ceiling(
                    min(int(arguments[extent]) for extent in extents)
                ),
            )

        accepted = [
            config
            for config in configs
            if all(
                config.kwargs[parameter] <= limit
                for parameter, limit in limits.items()
            )
            and (not config.kwargs.get("USE_TMA", 0) or descriptors_legal)
        ]
        if not accepted:
            raise ValueError(
                "no Triton autotune configuration satisfies the runtime "
                "extent constraints"
            )
        return accepted

    return {"early_config_prune": early_config_prune}


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
        "scan": (32, 64, 128, 256, 512, 1024),
        "stream_contract": (32, 64, 128),
        "stream_scaled": (1, 2, 4, 8),
        "query": (1, 2, 16, 32, 64, 128),
        "ragged_member": (32, 64, 128),
        "lane_pack": (64, 128, 256, 512),
        "pointwise_lane": (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384),
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
        "pointwise_lane",
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


def autotune_configurations(
    parameter_map: dict[str, str],
    extra_parameter_candidates: dict[str, tuple[int, ...]] | None = None,
    *,
    parameter_extents: dict[str, int] | None = None,
) -> list[object]:
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
        (
            ({"stream_scaled": 1}, 2, 4),
            ({"stream_scaled": 2}, 3, 4),
            ({"stream_scaled": 4}, 3, 8),
            ({"stream_scaled": 8}, 2, 8),
        ),
        (
            ({"pointwise_lane": 32}, 1, 1),
            ({"pointwise_lane": 64}, 1, 1),
            ({"pointwise_lane": 128}, 1, 4),
            ({"pointwise_lane": 256}, 1, 4),
            ({"pointwise_lane": 512}, 1, 4),
            ({"pointwise_lane": 1024}, 1, 8),
            ({"pointwise_lane": 2048}, 1, 8),
            ({"pointwise_lane": 4096}, 1, 8),
            ({"pointwise_lane": 8192}, 1, 8),
            ({"pointwise_lane": 16384}, 1, 8),
        ),
        (
            ({"program_m": 1, "pointwise_lane_n": 128}, 1, 4),
            ({"program_m": 1, "pointwise_lane_n": 256}, 1, 4),
            ({"program_m": 1, "pointwise_lane_n": 512}, 1, 4),
            ({"program_m": 1, "pointwise_lane_n": 1024}, 1, 8),
            ({"program_m": 2, "pointwise_lane_n": 256}, 1, 4),
            ({"program_m": 4, "pointwise_lane_n": 256}, 1, 4),
            ({"program_m": 8, "pointwise_lane_n": 256}, 1, 4),
            ({"program_m": 16, "pointwise_lane_n": 256}, 1, 4),
        ),
        (
            ({"pointwise_lane": 128, "program_n": 1}, 1, 4),
            ({"pointwise_lane": 256, "program_n": 1}, 1, 4),
            ({"pointwise_lane": 512, "program_n": 1}, 1, 4),
            ({"pointwise_lane": 1024, "program_n": 1}, 1, 8),
            ({"pointwise_lane": 256, "program_n": 2}, 1, 4),
            ({"pointwise_lane": 256, "program_n": 4}, 1, 4),
            ({"pointwise_lane": 256, "program_n": 8}, 1, 4),
            ({"pointwise_lane": 256, "program_n": 16}, 1, 4),
        ),
        tuple(
            (
                {
                    "program_m": m,
                    "query": n,
                    "stream_scaled": groups,
                    "group_m": 8,
                },
                stages,
                warps,
            )
            for m, n, groups, stages, warps in (
                (64, 16, 1, 4, 4),
                (64, 32, 1, 4, 4),
                (64, 32, 2, 4, 4),
                (64, 64, 1, 4, 4),
                (64, 64, 2, 4, 4),
                (64, 64, 4, 3, 4),
                (64, 128, 1, 4, 4),
                (64, 128, 2, 3, 4),
                (128, 16, 1, 4, 4),
                (128, 32, 1, 4, 4),
                (128, 32, 2, 4, 4),
                (128, 64, 1, 4, 4),
                (128, 64, 2, 3, 4),
                (128, 128, 1, 4, 4),
                (128, 128, 2, 3, 8),
            )
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
        tuple(
            (
                {"program_m": program, "stream_contract": stream},
                stages,
                warps,
            )
            for program, stream, stages, warps in (
                (32, 32, 4, 4),
                (64, 32, 4, 4),
                (64, 64, 3, 4),
                (64, 128, 3, 4),
                (128, 32, 4, 4),
                (128, 64, 3, 4),
                (128, 128, 2, 8),
            )
        ),
        tuple(
            ({"program_m": m, "program_n": n}, stages, warps)
            for m, n, stages, warps in (
                (8, 8, 5, 2),
                (8, 16, 5, 2),
                (16, 8, 5, 2),
                (16, 16, 4, 4),
                (16, 32, 3, 4),
                (32, 16, 3, 4),
                (32, 32, 2, 8),
                (16, 64, 2, 8),
                (64, 16, 2, 8),
                (32, 64, 3, 8),
                (64, 32, 3, 8),
                (64, 64, 2, 8),
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
            (64, 64, 256, 8, 3, 8),
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
    extras = extra_parameter_candidates or {}
    extra_names = tuple(extras)
    extra_values = tuple(product(*(extras[name] for name in extra_names)))
    if not extra_values:
        extra_values = ((),)
    extents = parameter_extents or {}
    configurations = []
    emitted = set()
    for values, stages, warps in choices:
        for combination in extra_values:
            target_values = _target_parameters(parameter_map, values)
            target_values.update(zip(extra_names, combination))
            for parameter, extent in extents.items():
                if parameter not in target_values:
                    raise ValueError(
                        f"Triton extent constraint names unknown parameter {parameter!r}"
                    )
                limit = _power_of_two_ceiling(extent)
                target_values[parameter] = min(target_values[parameter], limit)
            key = (tuple(sorted(target_values.items())), stages, warps)
            if key in emitted:
                continue
            emitted.add(key)
            configurations.append(
                triton.Config(
                    target_values,
                    num_stages=stages,
                    num_warps=warps,
                )
            )
    return configurations


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
