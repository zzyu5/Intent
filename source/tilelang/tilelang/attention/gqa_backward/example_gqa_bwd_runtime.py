import importlib.util
from pathlib import Path


def load_source():
    source_path = Path(__file__).with_name("example_gqa_bwd.py")
    spec = importlib.util.spec_from_file_location(
        "local_tilelang_gqa_backward", source_path
    )
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def main():
    root = next(parent for parent in Path(__file__).parents if parent.name == "tilelang" and parent.parent.name == "tilelang")
    support = root / "support" / "runtime_cases.py"
    spec = importlib.util.spec_from_file_location("intent_tilelang_runtime_cases", support)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    cases.run("gqa_backward", Path(__file__).with_name("example_gqa_bwd.py"))


if __name__ == "__main__":
    main()
