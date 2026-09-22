from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, q, k, v):
    return prepare_source(Path(__file__).with_name("linear.mojo"), "source_linear_attention",
        [view("q", (1, 2, 3)), view("k", (1, 2, 3)), view("v", (1, 2, 4)),
         view("output", (1, 2, 4), output=True), view("final_state", (1, 3, 4), output=True)],
        target, (q, k, v))
