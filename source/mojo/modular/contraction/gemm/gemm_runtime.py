from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, a, b):
    return prepare_source(Path(__file__).with_name("gemm.mojo"), "max_gemm",
        [view("a", (1, 3)), view("b", (3, 2)), view("c", (1, 2), output=True)],
        target, (a, b))
