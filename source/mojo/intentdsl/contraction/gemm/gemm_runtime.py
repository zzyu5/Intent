from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, a, b):
    if a.shape[0] % 64 or b.shape[1] % 64 or a.shape[1] % 32:
        raise NotImplementedError("this SIMD GEMM baseline requires M/N multiples of 64 and K multiple of 32")
    return prepare_source(Path(__file__).with_name("gemm.mojo"), "source_gemm",
        [view("a", (1, 3)), view("b", (3, 2)), view("c", (1, 2), output=True)],
        target, (a, b))
