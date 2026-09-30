"""Call the existing softmax algorithm from PyTorch using an installed Intent."""

import intent
import torch

from kernels.normalization.softmax import stable_softmax_f16


def main() -> None:
    softmax = intent.compile(stable_softmax_f16, target=intent.TritonTarget())
    x = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
    y = softmax.run(x)
    print(f"softmax: {tuple(y.shape)}, {y.dtype}, {y.device}")
    print(y[0, :8])
    print(f"Generated source and IR: {softmax.cache_directory}")


if __name__ == "__main__":
    main()
