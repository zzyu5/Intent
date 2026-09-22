from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, q, k, v, scale):
    return prepare_source(Path(__file__).with_name("causal.mojo"), "source_causal_attention",
        [view("q", (1, 2, 4)), view("k", (1, 3, 4)), view("v", (1, 3, 5)),
         view("output", (1, 2, 5), output=True), {"name": "scale", "kind": "scalar", "dtype": "f32"}],
        target, (q, k, v, scale))
