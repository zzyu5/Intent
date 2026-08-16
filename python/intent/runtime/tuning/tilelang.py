from __future__ import annotations

from itertools import product
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


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 256, 512, 1024),
        "scan": (32, 64, 128, 256, 512, 1024),
        "stream_contract": (32, 64, 128),
        "stream_scaled": (1, 2, 4, 8),
        "query": (1, 2, 64, 128),
        "ragged_member": (64, 128),
        "lane_pack": (64, 128, 256),
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
    raise NotImplementedError(f"unsupported TileLang tuner role: {role}")


def _completion_candidates(role: str) -> tuple[int, ...]:
    candidates = _role_candidates(role)
    if role == "query" or role.startswith("query_"):
        return tuple(value for value in candidates if value not in (1, 2))
    return candidates


def autotune_configurations(
    parameter_map: dict[str, str],
    *,
    equal_role_groups: tuple[tuple[str, ...], ...] = (),
    extra_parameters: dict[str, tuple[int, ...]] | None = None,
) -> list[dict[str, int]]:
    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    for group in equal_role_groups:
        if len(group) < 2 or not frozenset(group).issubset(roles):
            raise ValueError(
                "TileLang equal-role group must contain at least two configured roles"
            )
    joint_program_profiles = (
        (
            ({"program_m": 8, "program_n": 8}, 1, 128),
            ({"program_m": 16, "program_n": 16}, 1, 128),
            ({"program_m": 32, "program_n": 32}, 1, 128),
            ({"program_m": 128, "program_n": 64}, 2, 128),
            ({"program_m": 128, "program_n": 128}, 3, 256),
            ({"program_m": 64, "program_n": 128}, 2, 128),
        )
        if {"program_m", "program_n"}.issubset(roles)
        else ()
    )
    single_program_m_profiles = (
        (
            ({"program_m": 64}, 1, 128),
            ({"program_m": 64}, 2, 128),
            ({"program_m": 128}, 1, 128),
        )
        if roles == {"program_m"}
        else ()
    )
    profiles = (
        (
            ({"scan": 64}, 1, 128),
            ({"scan": 128}, 1, 128),
            ({"scan": 256}, 2, 256),
        ),
        (
            ({"stream": 256}, 1, 128),
            ({"stream": 512}, 1, 128),
            ({"stream": 1024}, 2, 256),
        ),
        (
            ({"stream_contract": 32}, 1, 128),
            ({"stream_contract": 64}, 1, 128),
        ),
        (
            ({"stream_scaled": 1}, 1, 128),
            ({"stream_scaled": 2}, 1, 128),
            ({"stream_scaled": 4}, 2, 128),
            ({"stream_scaled": 8}, 2, 256),
        ),
        tuple(
            ({"query": query, stream_role: stream}, num_stages, threads)
            for stream_role in ("stream", "stream_contract")
            for query, stream, num_stages, threads in (
                (1, 32, 1, 128),
                (2, 32, 1, 128),
                (64, 64, 1, 128),
                (64, 64, 2, 128),
                (128, 64, 1, 128),
                (128, 128, 1, 128),
            )
        ),
        joint_program_profiles,
        single_program_m_profiles,
        (
            ({"program_m": 128, "program_n": 64, "reduction": 64, "group_m": 8}, 2, 128),
            ({"program_m": 128, "program_n": 128, "reduction": 32, "group_m": 8}, 3, 256),
            ({"program_m": 64, "program_n": 128, "reduction": 64, "group_m": 8}, 2, 128),
        ),
        (
            ({"ragged_member": 128, "feature": 64, "reduction": 64}, 2, 128),
            ({"ragged_member": 128, "feature": 128, "reduction": 32}, 3, 256),
            ({"ragged_member": 64, "feature": 128, "reduction": 64}, 2, 128),
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
    for index, (profile, num_stages, threads) in enumerate(selected):
        values = {
            role: profile.get(
                role,
                _completion_candidates(role)[
                    index % len(_completion_candidates(role))
                ],
            )
            for role in roles
        }
        if any(
            len({values[role] for role in group}) != 1
            for group in equal_role_groups
        ):
            continue
        key = (tuple(sorted(values.items())), num_stages, threads)
        if key not in seen:
            seen.add(key)
            choices.append((values, num_stages, threads))
    if not choices:
        raise NotImplementedError(
            "TileLang has no legal autotuning configuration for the requested role constraints"
        )
    configurations = [
        _configuration(
            parameter_map,
            values,
            num_stages=num_stages,
            threads=threads,
        )
        for values, num_stages, threads in choices
    ]
    if not extra_parameters:
        return configurations
    names = tuple(extra_parameters)
    candidates = tuple(extra_parameters[name] for name in names)
    if any(not values for values in candidates):
        raise ValueError("TileLang extra tuner parameters require candidates")
    return [
        {**configuration, **dict(zip(names, values))}
        for configuration in configurations
        for values in product(*candidates)
    ]


def row_autotune_configurations() -> list[dict[str, int]]:
    return [
        {"num_stages": num_stages, "threads": threads}
        for num_stages, threads in ((1, 128), (2, 128), (1, 256), (2, 256))
    ]
