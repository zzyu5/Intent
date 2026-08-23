from __future__ import annotations

from .. import implementation_gap


CASES = {
    "nvfp4_quantize": implementation_gap(
        "the source contract quantizes bf16 groups of 16 with one runtime global "
        "encoding scale, stores E2M1 pairs as bytes, and stores E4M3 scales in a "
        "fixed 128x4-to-512-byte tile swizzle; Intent has neither typed E2M1 "
        "packing nor a first-class swizzled scale-storage ABI"
    ),
}
