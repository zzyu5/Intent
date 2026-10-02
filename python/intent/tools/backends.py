"""Public setup descriptions; compilation still uses the existing Target API."""

BACKENDS = {
    "triton": {
        "target": "TritonTarget", "modules": ("torch", "triton"),
        "requirements": ("triton==3.6.0", "numpy==1.26.4"), "torch": "2.10.0",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "python_max": (3, 12), "toolchain": "NVIDIA driver and CUDA-capable PyTorch",
    },
    "cutile": {
        "target": "CuTileTarget", "modules": ("torch", "cuda.tile", "cuda.tile.tune"),
        "requirements": ("triton==3.6.0", "numpy==1.26.4",
                         "cuda-toolkit[tileiras,nvvm,nvcc]==13.3.1", "cuda-tile==1.6.0"),
        "torch": "2.10.0",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "python_max": (3, 12), "toolchain": "NVIDIA driver and the CUDA tile compiler",
    },
    "mojo": {
        "target": "MojoTarget", "modules": ("torch",),
        "requirements": (), "torch": "2.10.0",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "python_max": (3, 12), "toolchain": "An explicitly installed Mojo compiler; Linux x86-64 AVX2/AVX512",
    },
    "weft": {
        "target": "WeftTarget", "modules": (),
        "requirements": (), "torch": None, "torch_index": None,
        "python_max": (3, 12), "toolchain": "An Intent compiler built with Weft; public generation requires explicit vector_bits and workers",
    },
    "bangc": {
        "target": "BangCTarget", "modules": (),
        "requirements": (), "torch": None, "torch_index": None,
        "python_max": (3, 12), "toolchain": "An explicitly installed NeuWare SDK and MLU370 runtime",
    },
}


def backend(name: str) -> dict:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[name]


def make_target(name: str, options: dict, *, facts: dict | None = None):
    import intent
    from intent.targets.specification import read_compilation_target

    if facts is not None:
        if options:
            raise ValueError("explicit compiler facts do not accept local runtime target options")
        return read_compilation_target(name, facts)
    return getattr(intent, backend(name)["target"])(**options)
