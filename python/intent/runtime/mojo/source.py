from pathlib import Path

from .program import NativeProgram


def view(name: str, dimensions: tuple[int, ...], *, output: bool = False) -> dict[str, object]:
    return {"name": name, "kind": "view", "access": 1 if output else 0,
            "shape": [-1] * len(dimensions), "dimensions": list(dimensions),
            "alias": "", "noalias": False}


def scalar(name: str) -> dict[str, object]:
    return {"name": name, "kind": "scalar", "dtype": "f32"}


def prepare_source(path: Path, entry: str, parameters, target, arguments):
    if target.workers != 8:
        raise NotImplementedError("the initial Mojo source corpus uses an explicit eight-worker budget")
    metadata = {"parameters": parameters, "workers": target.workers,
                "contiguous_views": True, "disjoint_outputs": True,
                "candidates": [{"entry": entry, "values": []}]}
    return NativeProgram(path.read_text(encoding="utf-8"), metadata, target).prepare(arguments)
