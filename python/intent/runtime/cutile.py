from .artifact import CompiledArtifact
from .source import materialize_python_source


def array_index_kernels(function, view_names: tuple[str, ...]):
    from types import FunctionType
    import cuda.tile as ct

    narrow = FunctionType(function.__code__, function.__globals__, function.__name__,
                          function.__defaults__, function.__closure__)
    narrow.__qualname__ = function.__qualname__
    narrow.__annotations__ = dict(function.__annotations__)
    for name in view_names:
        narrow.__annotations__[name] = ct.Array
    return ct.kernel(narrow), ct.kernel(function)


def can_use_i32_array_indices(views: tuple, tile_bounds: tuple) -> bool:
    limit = (1 << 31) - 1
    for view, tiles in zip(views, tile_bounds, strict=True):
        span = 0
        padded_elements = 1
        for extent, stride, tile in zip(view.shape, view.stride(), tiles, strict=True):
            if extent <= 0 or tile <= 0 or stride < 0 or stride > limit:
                return False
            # Native access origins are in the view; include the physical tail.
            padded_extent = extent + tile - 1
            if padded_extent > limit:
                return False
            span += (padded_extent - 1) * stride
            padded_elements *= padded_extent
        if (span + 1) * view.element_size() > limit or padded_elements > limit:
            return False
    return True


def materialize_cutile_artifact(
    source: str,
    module_text: str,
    entry_name: str,
    device: int,
) -> CompiledArtifact:
    return materialize_python_source(
        target_name="cutile",
        source=source,
        module_text=module_text,
        entry_name=entry_name,
        device=device,
        backend_ir_collector=None,
    )
