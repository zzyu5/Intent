from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path
import re
import subprocess


def snapshot(project: Path) -> dict:
    """Freeze only public author material, never benchmarks or provider sources."""
    import intent
    import intent.language as language
    from intent.language.builtins import INTRINSICS, Intrinsic, IntrinsicNamespace, QuantFormats
    from intent.language.signatures import INTRINSIC_SIGNATURES

    documents = {}
    for directory in (project / "doc/dsl", project / "doc/programming-model"):
        for path in sorted(directory.rglob("*")):
            if path.suffix not in {".md", ".py"}:
                continue
            identifier = str(path.relative_to(project))
            text = path.read_text()
            sections = list(re.finditer(r"^#{1,3} (.+)$", text, re.MULTILINE)) if path.suffix == ".md" else []
            documents[identifier] = {
                "id": identifier, "title": sections[0][1] if sections else path.stem,
                "kind": "example" if path.suffix == ".py" else "concept",
                "source": identifier, "line": 1, "text": text,
            }
            for index, section in enumerate(sections):
                end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
                line = text.count("\n", 0, section.start()) + 1
                section_id = f"{identifier}#L{line}"
                documents[section_id] = {
                    "id": section_id, "title": section[1], "kind": "concept",
                    "source": identifier, "line": line, "text": text[section.start():end],
                }

    exports = {name: getattr(language, name) for name in language.__all__}
    exports.update(INTRINSICS)
    exports.update({f"intent.{name}": getattr(intent, name) for name in intent.__all__})
    for name, value in list(exports.items()):
        if inspect.isclass(value):
            exports.update({f"{name}.{member}": method for member, method in vars(value).items()
                            if inspect.isfunction(method) and (not member.startswith("_") or member == "__call__")})
        elif isinstance(value, QuantFormats):
            exports.update({f"{name}.{member}": format for member, format in vars(value).items()
                            if not member.startswith("_")})
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
            source = str(Path(inspect.getfile(value)).relative_to(project))
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
        """Find API names, concepts, diagnostics and examples. Read returned IDs for full context."""
        if kind not in {"all", "api", "concept", "diagnostic", "example"}:
            raise ValueError("kind must be all, api, concept, diagnostic or example")
        terms = re.findall(r"[\w.]+", query.lower())
        if not terms:
            raise ValueError("query must contain a name or search term")
        results = []
        if kind in {"all", "api"}:
            for name, entry in self.corpus["symbols"].items():
                score = sum(term in name.lower() for term in terms)
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
        """Exact public API declaration plus authoritative rules, return schema and examples."""
        name = name.removeprefix("intent.language.").removeprefix("I.")
        entry = self.corpus["symbols"].get(name)
        if entry is None:
            return {"status": "not-found", "name": name, "message": "Not a current public declaration; no replacement is inferred."}
        return {"status": "declared", "revision": self.corpus["revision"], **entry,
                "signature_note": None if entry["signature"] else "No inspectable signature is declared; consult the linked rules, not a guessed signature.",
                "rules": [self.corpus["documents"][key] for key in entry["sections"]
                          if self.corpus["documents"][key]["kind"] == "concept"],
                "implementation_diagnostics": [self.corpus["documents"][key] for key in entry["sections"]
                                               if self.corpus["documents"][key]["kind"] == "diagnostic"],
                "examples": [key for key in entry["sections"] if self.corpus["documents"][key]["kind"] == "example"],
                "verification": "not evaluated by this read-only service; diagnostics do not redefine doc semantics"}

    def read(self, id: str, section: str | None = None) -> dict:
        """Read a published document/example ID, optionally an exact section title. No filesystem paths accepted."""
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
    server = FastMCP("intent_manual", instructions="Intent public manual. Query api for exact signatures and return rules; read complete examples. No execution or task answers.")
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    for method in (manual.search, manual.api, manual.read):
        server.add_tool(method, annotations=annotations)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
