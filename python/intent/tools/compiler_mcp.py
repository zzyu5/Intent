"""Explicitly enabled compilation tools, separate from the read-only manual."""
from pathlib import Path

from .compilation import compile_request, doctor, generate_ir_request, materialize_request, optimize_request


def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        if error.name != "mcp":
            raise
        raise SystemExit("Install IntentDSL with its 'manual' extra to use the MCP servers.") from error

    server = FastMCP("intent_compiler", instructions=(
        "Use only existing Python program, IR or saved program paths explicitly supplied by the user. "
        "Loading that module executes its top-level Python host code. "
        "These tools use the public Intent pipeline and do not themselves launch the selected kernel. "
        "Generated/materialized does not mean numerical or performance validation. "
        "Use the separate intent_manual server for language contracts."
    ))

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def compile(program_path: str, kernel: str, target: str | None = None,
                      target_options: dict | None = None, constexprs: dict | None = None,
                      compiler: str | None = None, tuning_config: str | None = None,
                      materialize: bool = False, stage: str = "provider",
                      target_facts: dict | None = None, export_directory: str | None = None,
                      options: dict | None = None) -> dict:
        """Compile an existing .py file and report stages/artifacts.

        stage='kir' needs no target, provider SDK or device. 'shared' and
        'provider' need a target; only 'provider' permits materialize=True.
        target_facts accepts the compiler's explicit target object for offline
        generation. export_directory saves source, final IR and metadata for a
        later host; target_options select the local runtime if materializing.
        The file's ordinary top-level Python host code executes normally.
        options accepts numerics ('source' or 'relaxed_normalization'),
        online_reduction (bool), and optimization_remarks (bool). Source includes
        the language's FMA/reduction permissions, not bitwise reproducibility.
        Relaxed normalization permits normalized-summary rescaling and movement
        of low-precision weight casts. Valid-member scores and all values entering
        the moment contraction must be finite, including zero-weight terms;
        zero times infinity is not an inactive access. Masked accesses keep fills.
        It can change rounding/underflow/overflow but does not enable global fast
        math or FTZ. The online switch controls optimization, not permission.
        """
        path = Path(program_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix != ".py":
            raise ValueError("program_path must name an existing Python file supplied by the user")
        # Run synchronously in the server event loop: module loading temporarily
        # owns sys.path/stdout, so concurrent compilation requests must not overlap.
        return compile_request(str(path), kernel, target, target_options=target_options,
                               constexprs=constexprs, compiler=compiler,
                               tuning_config=tuning_config, materialize=materialize, stage=stage,
                               target_facts=target_facts, export_directory=export_directory,
                               options=options)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def generate_from_ir(ir_file: str, name: str, target: str,
                               input_stage: str = "shared", target_options: dict | None = None,
                               compiler: str | None = None, materialize: bool = False,
                               target_facts: dict | None = None, export_directory: str | None = None) -> dict:
        """Generate provider source from explicit existing KIR or shared IR.

        Supply the input stage, target and diagnostic program name explicitly.
        The whole module is compiled; name does not select a kernel. Callable
        entries and candidates are determined by the IR and its metadata.
        Shared input retains its physical program/configuration and must agree
        with the selected target's capabilities. Its compile options are preserved
        from IR, not reselected by this tool. This does not launch a kernel.
        """
        return generate_ir_request(ir_file, name, target, input_stage=input_stage,
                                   target_options=target_options, compiler=compiler,
                                   materialize=materialize, target_facts=target_facts,
                                   export_directory=export_directory)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def materialize(program_directory: str, target: str, target_options: dict | None = None) -> dict:
        """Load a generated program from an explicit directory and bind a matching local runtime.

        Target capabilities must agree with the saved compiler facts. This may
        compile/load native code, but neither recompiles KIR nor launches a kernel.
        """
        return materialize_request(program_directory, target, target_options=target_options)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def optimize(ir_file: str, pipeline: str, optimizer: str | None = None) -> dict:
        """Run a standard MLIR pipeline on an existing IR file supplied by the user.

        The installed intent-opt applies the requested passes and reports output
        and diagnostic paths. Pass prerequisites belong to the input/current IR;
        this does not materialize or launch a kernel or establish correctness.
        """
        return optimize_request(ir_file, pipeline, optimizer=optimizer)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    async def environment(target: str, target_options: dict | None = None,
                          compiler: str | None = None, target_facts: dict | None = None) -> dict:
        """Inspect the selected compiler/target; explicit facts skip local SDK/device probing.

        Without explicit facts, also inspect selected runtime dependencies. This
        does not establish numerical correctness or device execution.
        """
        return doctor(target, target_options=target_options, compiler=compiler, target_facts=target_facts)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
