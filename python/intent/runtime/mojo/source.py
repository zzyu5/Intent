from pathlib import Path

from .program import NativeProgram


def view(name: str, dimensions: tuple[int, ...], *, output: bool = False) -> dict[str, object]:
    return {"name": name, "kind": "view", "dtype": "f32", "access": 1 if output else 0,
            "shape": [-1] * len(dimensions), "dimensions": list(dimensions),
            "strides": [None] * len(dimensions), "alias": "", "noalias": False}


def scalar(name: str) -> dict[str, object]:
    return {"name": name, "kind": "scalar", "dtype": "f32"}


def prepare_source(path: Path, entry: str, parameters, target, arguments):
    if target.workers != 8:
        raise NotImplementedError("the initial Mojo source corpus uses an explicit eight-worker budget")
    source = path.read_text(encoding="utf-8")
    metadata = {"parameters": parameters, "workers": target.workers,
                "contiguous_views": True, "disjoint_outputs": True,
                "source_prelude_end": 0,
                "candidates": [{"entry": entry, "values": [],
                                "source_range": [0, len(source.encode("utf-8"))]}]}
    return NativeProgram(source, metadata, target).prepare(arguments)
