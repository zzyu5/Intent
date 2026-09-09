from pathlib import Path
from intent.runtime.mojo.source import prepare_source, scalar, view


def prepare(target, x, weight, inverse_features, epsilon):
    if inverse_features != 1.0 / x.shape[-1]:
        raise NotImplementedError("MAX RMSNorm uses the reciprocal of the normalized extent")
    return prepare_source(Path(__file__).with_name("rms_norm.mojo"), "max_rms_norm",
        [view("x", (1, 2)), view("weight", (2,)), view("y", (1, 2), output=True),
         scalar("inverse_features"), scalar("epsilon")], target,
        (x, weight, inverse_features, epsilon))
