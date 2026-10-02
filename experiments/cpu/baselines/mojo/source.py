from pathlib import Path
from dataclasses import dataclass

from intent.runtime.mojo.program import NativeProgram


@dataclass(frozen=True)
class SourceParameter:
    """A hand-written corpus entry's declared public parameter and C ABI slots."""

    declaration: dict
    slots: tuple[dict, ...]


def view(name: str, dimensions: tuple[int, ...], *, output: bool = False) -> SourceParameter:
    return SourceParameter(
        {"name": name, "kind": "view", "dtype": "f32", "access": 1 if output else 0,
         "shape": [-1] * len(dimensions), "dimensions": list(dimensions),
         "strides": [None] * len(dimensions), "alias": "", "noalias": False},
        ({"role": "pointer", "carrier": "ptr", "element": "f32"},
         *({"role": "extent", "axis": axis, "carrier": "i64"} for axis in range(len(dimensions))),
         *({"role": "stride", "axis": axis, "carrier": "i64"} for axis in range(len(dimensions)))),
    )


def scalar(name: str) -> SourceParameter:
    return SourceParameter({"name": name, "kind": "scalar", "dtype": "f32"},
                           ({"role": "scalar", "carrier": "f32"},))


def prepare_source(path: Path, entry: str, parameters, target, arguments):
    if target.workers != 8:
        raise NotImplementedError("the initial Mojo source corpus uses an explicit eight-worker budget")
    source = path.read_text(encoding="utf-8")
    views = [(position, parameter.declaration) for position, parameter in enumerate(parameters)
             if parameter.declaration["kind"] == "view"]
    requirements = {
        "views": [{"parameter": position, "layout": "contiguous", "alignment": 1}
                  for position, _ in views],
        "disjoint": [{"left": left, "right": right}
                     for index, (left, lhs) in enumerate(views)
                     for right, rhs in views[index + 1:]
                     if lhs["access"] != 0 or rhs["access"] != 0],
    }
    metadata = {"interface": {"parameters": [parameter.declaration for parameter in parameters]},
                "native": {"requirements": requirements,
                           "slots": [{**slot, "parameter": position} for position, parameter in enumerate(parameters)
                                     for slot in parameter.slots]},
                "source_prelude_end": 0,
                "candidates": [{"entry": entry, "values": [],
                                "source_range": [0, len(source.encode("utf-8"))]}]}
    return NativeProgram(source, metadata, target).prepare(arguments)
