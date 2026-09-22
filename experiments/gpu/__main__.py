import os
import sys


if cutile_site := os.environ.get("INTENT_CUTILE_SITE_PACKAGES"):
    # Keep the source environment's Torch/Triton when adding the other backend.
    import torch
    import triton

    sys.path.insert(0, cutile_site)

from experiments._common.runner import main


if __name__ == "__main__":
    main(providers=("triton", "cutile", "tilelang"))
