import importlib.util
from pathlib import Path


def main():
    support = Path(__file__).parents[1] / "support" / "projection_runtime.py"
    spec = importlib.util.spec_from_file_location("intent_moe_projection_runtime", support)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(Path(__file__).with_name("v0_moe_fused.py"), "grouped")


if __name__ == "__main__":
    main()
