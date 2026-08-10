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


def _role_candidates(role: str) -> tuple[int, ...]:
    candidates = {
        "stream": (32, 64, 128, 256, 512, 1024),
        "stream_contract": (32, 64, 128),
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
    raise NotImplementedError(f"unsupported TileLang tuner role: {role}")


def autotune_configurations(parameter_map: dict[str, str]) -> list[dict[str, int]]:
    roles = frozenset(parameter_map.values())
    for role in roles:
        _role_candidates(role)
    profiles = (
        (
            ({"stream": 256}, 1, 128),
            ({"stream": 512}, 1, 128),
            ({"stream": 1024}, 2, 256),
        ),
        (
            ({"stream_contract": 32}, 1, 128),
            ({"stream_contract": 64}, 1, 128),
        ),
        tuple(
            ({"query": query, stream_role: stream}, num_stages, threads)
            for stream_role in ("stream", "stream_contract")
            for query, stream, num_stages, threads in (
                (64, 64, 1, 128),
                (64, 64, 2, 128),
                (128, 64, 1, 128),
                (128, 128, 1, 128),
            )
        ),
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
            role: profile.get(role, _role_candidates(role)[index % len(_role_candidates(role))])
            for role in roles
        }
        key = (tuple(sorted(values.items())), num_stages, threads)
        if key not in seen:
            seen.add(key)
            choices.append((values, num_stages, threads))
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
