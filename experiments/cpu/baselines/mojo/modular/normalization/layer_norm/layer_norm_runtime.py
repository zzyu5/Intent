from pathlib import Path
from intent.runtime.mojo.source import prepare_source, scalar, view


def prepare(target, x, weight, bias, inverse_features, epsilon):
    return prepare_source(Path(__file__).with_name("layer_norm.mojo"), "max_layer_norm_moments",
        [view("x", (1, 2)), view("weight", (2,)), view("bias", (2,)),
         view("y", (1, 2), output=True), scalar("inverse_features"), scalar("epsilon")],
        target, (x, weight, bias, inverse_features, epsilon))
