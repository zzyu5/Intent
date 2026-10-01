from pathlib import Path
from experiments.cpu.baselines.mojo.source import prepare_source, scalar, view


def prepare(target, q, k, v, scale):
    return prepare_source(Path(__file__).with_name("causal.mojo"), "source_causal_attention",
        [view("q", (1, 2, 4)), view("k", (1, 3, 4)), view("v", (1, 3, 5)),
         view("output", (1, 2, 5), output=True), scalar("scale")],
        target, (q, k, v, scale))
