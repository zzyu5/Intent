from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, x, scale, bias):
    return prepare_source(Path(__file__).with_name("batched_affine.mojo"), "max_affine",
        [view("x", (1, 2, 3)), view("scale", (1, 2)), view("bias", (1, 2)),
         view("output", (1, 2, 3), output=True)], target, (x, scale, bias))
