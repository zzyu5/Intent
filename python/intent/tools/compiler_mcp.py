"""Explicitly enabled compilation tools, separate from the read-only manual."""
from pathlib import Path

from .compilation import compile_request, doctor


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise SystemExit("Install IntentDSL with its 'manual' extra to use the MCP servers.") from error

    server = FastMCP("intent_compiler", instructions=(
        "Compile only an existing Python program path explicitly supplied by the user. "
        "Loading that module executes its top-level Python host code. "
        "These tools use the public Intent pipeline and do not themselves launch the selected kernel. "
        "Generated/materialized does not mean numerical or performance validation. "
        "Use the separate intent_manual server for language contracts."
    ))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def compile(program_path: str, kernel: str, target: str,
                      target_options: dict | None = None, constexprs: dict | None = None,
                      compiler: str | None = None, tuning_config: str | None = None,
                      materialize: bool = False) -> dict:
        """Compile an existing .py file and report stages/artifacts. Its top-level host code executes normally."""
        path = Path(program_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("program_path must name an existing Python file supplied by the user")
        # Run synchronously in the server event loop: module loading temporarily
        # owns sys.path/stdout, so concurrent compilation requests must not overlap.
        return compile_request(str(path), kernel, target, target_options=target_options,
                               constexprs=constexprs, compiler=compiler,
                               tuning_config=tuning_config, materialize=materialize)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def environment(target: str, target_options: dict | None = None,
                          compiler: str | None = None) -> dict:
        """Inspect only the selected backend's dependencies and target facts; this is not a numerical check."""
        return doctor(target, target_options=target_options, compiler=compiler)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
