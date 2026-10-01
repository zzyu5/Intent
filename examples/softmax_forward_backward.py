"""Compose existing forward/backward kernels explicitly in ordinary Python."""
import argparse

import intent
import torch

from kernels.backward.softmax import ROWS, COLUMNS, softmax_backward
from kernels.normalization.softmax import stable_softmax


def make_softmax_pair(target):
    # Compile once; these artifacts belong to the host wrapper's lifetime.
    forward = intent.compile(stable_softmax, target=target)
    backward = intent.compile(softmax_backward, target=target)

    def forward_backward(x, upstream):
        probabilities = forward.run(x)
        gradient = backward.run(probabilities, upstream)
        return probabilities, gradient

    return forward_backward, (forward, backward)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("triton", "cutile"), default="triton")
    args = parser.parse_args()
    target = intent.TritonTarget() if args.target == "triton" else intent.CuTileTarget()
    program, artifacts = make_softmax_pair(target)
    x = torch.randn((ROWS, COLUMNS), dtype=torch.float32, device="cuda")
    upstream = torch.randn_like(x)
    probabilities, gradient = program(x, upstream)
    print(f"probabilities: {probabilities.shape}; gradient: {gradient.shape}")
    for artifact in artifacts:
        print(f"Compiler artifacts: {artifact.cache_directory}")


if __name__ == "__main__":
    main()
