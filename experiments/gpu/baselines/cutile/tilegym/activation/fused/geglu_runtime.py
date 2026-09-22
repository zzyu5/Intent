import importlib.util
from pathlib import Path


def main():
    root = next(parent for parent in Path(__file__).parents if parent.name == "tilegym")
    support = root / "support" / "runtime_cases.py"
    spec = importlib.util.spec_from_file_location("intent_cutile_runtime_cases", support)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)
    cases.run("geglu", Path(__file__).with_name("geglu.py"))


if __name__ == "__main__":
    main()
