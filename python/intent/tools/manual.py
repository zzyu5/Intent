from __future__ import annotations

import argparse
import ast
from dataclasses import fields
import inspect
import json
from pathlib import Path
import re
import subprocess


def _sections(text: str) -> list[tuple[int, int, int, str]]:
    sections = []
    fence = ""
    offset = 0
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        content = line.rstrip("\r\n")
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", content)
        if fence:
            if (marker and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence) and not marker[2].strip()):
                fence = ""
        elif marker and not (marker[1][0] == "`" and "`" in marker[2]):
            fence = marker[1]
        else:
            heading = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", content)
            if heading:
                title = re.sub(r"[ \t]+#+$", "", heading[2])
                sections.append((offset, number, len(heading[1]), title))
        offset += len(line)
    return sections


def snapshot(project: Path) -> dict:
    """Freeze language and GPU execution contracts, without algorithm examples."""
    project = project.resolve()
    import intent
    import intent.language as language
    from intent.language.builtins import INTRINSICS, Intrinsic, IntrinsicNamespace, QuantFormats
    from intent.language.signatures import INTRINSIC_SIGNATURES

    documents = {}
    paths = [path for directory in (project / "doc/dsl", project / "doc/programming-model")
             for path in directory.glob("*.md")]
    paths.append(project / "doc/compiler/kir-to-gpu.md")
    for path in sorted(paths):
        identifier = str(path.relative_to(project))
        text = path.read_text()
        sections = _sections(text)
        documents[identifier] = {
            "id": identifier, "title": sections[0][3] if sections else path.stem,
            "kind": "concept",
            "source": identifier, "line": 1, "text": text,
        }
        for index, (start, line, level, title) in enumerate(sections):
            end = next((start for start, _, depth, _ in sections[index + 1:]
                        if depth <= level), len(text))
            section_id = f"{identifier}#L{line}"
            documents[section_id] = {
                "id": section_id, "title": title,
                "kind": "concept",
                "source": identifier, "line": line, "text": text[start:end],
            }

    exports = {name: getattr(language, name) for name in language.__all__}
    exports.update(INTRINSICS)
    exports.update({f"intent.{name}": getattr(intent, name) for name in intent.__all__})
    for name, value in list(exports.items()):
        if inspect.isclass(value):
            exports.update({f"{name}.{member}": method for member, method in vars(value).items()
                            if inspect.isfunction(method) and (not member.startswith("_") or member == "__call__")})
        elif isinstance(value, QuantFormats):
            exports.update({f"{name}.{field.name}": getattr(value, field.name)
                            for field in fields(value) if not field.name.startswith("_")})
    for path in sorted((project / "python/intent/frontend/lowering").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            is_diagnostic = isinstance(node.func, ast.Attribute) and node.func.attr == "error"
            is_unsupported = isinstance(node.func, ast.Name) and node.func.id == "NotImplementedError"
            if not (is_diagnostic or is_unsupported):
                continue
            message = node.args[-1]
            if not isinstance(message, ast.Constant) or not isinstance(message.value, str):
                continue
            source = str(path.relative_to(project))
            identifier = f"diagnostic:{source}:L{node.lineno}"
            documents[identifier] = {
                "id": identifier, "title": message.value, "kind": "diagnostic",
                "source": source, "line": node.lineno,
                "text": "Current implementation diagnostic, not a language restriction or proof that its guard applies:\n" + message.value,
            }
    symbols = {}
    for name, value in exports.items():
        if isinstance(value, (Intrinsic, IntrinsicNamespace)):
            signature = INTRINSIC_SIGNATURES.get(name)
            source = "python/intent/language/signatures.py" if signature else "python/intent/language/builtins.py"
        elif inspect.isfunction(value) or inspect.isclass(value):
            signature = inspect.signature(value) if inspect.isfunction(value) or inspect.isfunction(vars(value).get("__init__")) else None
            source = str(Path(inspect.getfile(value)).resolve().relative_to(project))
        else:
            signature, source = None, "python/intent/language/__init__.py"
        pattern = re.compile(r"(?<![\w.])(?:I\.)?" + re.escape(name) + r"(?![\w.])")
        references = [d["id"] for d in documents.values()
                      if ("#L" in d["id"] or d["kind"] != "concept") and pattern.search(d["text"])]
        symbols[name] = {
            "name": name, "signature": str(signature) if signature else None,
            "declaration": source, "sections": references,
            "members": list(value.members) if isinstance(value, IntrinsicNamespace) else [],
            "availability": "public declaration; backend support and performance are not implied",
        }
    return {
        "revision": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
        "documents": documents, "symbols": symbols,
    }


class Manual:
    def __init__(self, corpus: dict):
        self.corpus = corpus

    def search(self, query: str, kind: str = "all") -> dict:
        """Find API names, language rules and diagnostics. Read returned IDs for full context."""
        if kind not in {"all", "api", "concept", "diagnostic"}:
            raise ValueError("kind must be all, api, concept or diagnostic")
        terms = re.findall(r"[\w.]+", query.lower())
        if not terms:
            raise ValueError("query must contain a name or search term")
        results = []
        if kind in {"all", "api"}:
            api_terms = [term.removeprefix("intent.language.").removeprefix("i.")
                         for term in terms]
            for name, entry in self.corpus["symbols"].items():
                score = sum(bool(term) and term in name.lower() for term in api_terms)
                if score:
                    results.append((100 * score, {"id": name, "kind": "api", "title": name,
                                                  "signature": entry["signature"]}))
        for entry in self.corpus["documents"].values():
            if kind not in {"all", entry["kind"]}:
                continue
            if "#L" not in entry["id"] and entry["kind"] == "concept":
                continue
            body, title = entry["text"].lower(), entry["title"].lower()
            score = sum(5 * (term in title) + (term in body) for term in terms)
            if score:
                results.append((score, {key: entry[key] for key in ("id", "title", "kind", "source", "line")}))
        results.sort(key=lambda row: (-row[0], row[1]["id"]))
        return {"revision": self.corpus["revision"], "matches": [r[1] for r in results[:12]],
                "total": len(results)}

    def api(self, name: str) -> dict:
        """Call api(name=...) for an exact declaration and rule IDs; use read(id=...) for rule text."""
        name = name.removeprefix("intent.language.").removeprefix("I.")
        entry = self.corpus["symbols"].get(name)
        if entry is None:
            return {"status": "not-found", "name": name, "message": "Not a current public declaration; no replacement is inferred."}
        return {"status": "declared", "revision": self.corpus["revision"],
                **{key: value for key, value in entry.items() if key != "sections"},
                "signature_note": None if entry["signature"] else "No inspectable signature is declared; consult the linked rules, not a guessed signature.",
                "rules": [{field: self.corpus["documents"][key][field]
                           for field in ("id", "title", "source", "line")}
                          for key in entry["sections"]
                          if self.corpus["documents"][key]["kind"] == "concept"],
                "read_note": "Read the relevant rule IDs for return shapes, dtypes and semantics. Implementation diagnostics are available through search(kind='diagnostic').",
                "verification": "not evaluated by this read-only service; diagnostics do not redefine doc semantics"}

    def read(self, id: str, section: str | None = None) -> dict:
        """Read a published rule ID, optionally an exact section title. No filesystem paths accepted."""
        if section is not None:
            entries = [d for d in self.corpus["documents"].values()
                       if d["source"] == id and d["title"] == section and "#L" in d["id"]]
            if len(entries) != 1:
                raise ValueError("section title must match exactly one section in this document")
            entry = entries[0]
        else:
            if id not in self.corpus["documents"]:
                raise ValueError("unknown published document ID; use search")
            entry = self.corpus["documents"][id]
        return {"revision": self.corpus["revision"], **entry}


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Intent author manual MCP (requires mcp>=1.28,<2)")
    parser.add_argument("--corpus", type=Path, required=True, help="Frozen public material JSON, not a repository root")
    arguments = parser.parse_args()
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    manual = Manual(json.loads(arguments.corpus.read_text()))
    server = FastMCP("intent_manual", instructions=(
        "Intent language and GPU execution contracts. Call api(name='I.domain') for exact declarations and rule IDs; "
        "read(id=...) for syntax, types, semantics and callable interface rules. "
        "read(id=..., section=...) accepts an exact section title. "
        "Use search(query=..., kind=...) to find names and IDs. "
        "No execution or task answers."
    ))
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    for method in (manual.search, manual.api, manual.read):
        server.add_tool(method, annotations=annotations)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
