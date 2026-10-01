"""Call the existing softmax algorithm from PyTorch using an installed Intent."""

import argparse
import intent
import torch

from kernels.normalization.softmax import stable_softmax_f16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("triton", "cutile"), default="triton")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Call through the opaque PyTorch operator adapter")
    args = parser.parse_args()
    target = intent.TritonTarget() if args.target == "triton" else intent.CuTileTarget()
    softmax = intent.compile(stable_softmax_f16, target=target)
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    if args.torch_compile:
        operation = softmax.as_torch_op("intent_examples::softmax")
        # Prepare the same real invocation before graph capture so first-use
        # provider compilation and tuning happen outside the captured graph.
        operation(x)

        def apply(value):
            return operation(value)

        y = torch.compile(apply, fullgraph=True)(x)
    else:
        y = softmax.run(x)
    print(f"softmax: {tuple(y.shape)}, {y.dtype}, {y.device}")
    print(y[0, :8])
    print(f"Generated source and IR: {softmax.cache_directory}")


if __name__ == "__main__":
    main()
