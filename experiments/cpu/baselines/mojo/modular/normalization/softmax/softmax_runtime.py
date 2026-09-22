from pathlib import Path
from intent.runtime.mojo.source import prepare_source, view


def prepare(target, x):
    return prepare_source(Path(__file__).with_name("softmax.mojo"), "max_softmax",
        [view("x", (1, 2)), view("y", (1, 2), output=True)], target, (x,))
