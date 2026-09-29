from __future__ import annotations

import argparse
import ast
from dataclasses import fields
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Literal


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


def _section_body(text: str) -> str:
    """Index a heading's own text without repeating its child sections."""
    sections = _sections(text)
    return text[:sections[1][0]] if len(sections) > 1 else text


def snapshot(project: Path) -> dict:
    """Freeze public language and kernel/host contracts, without algorithm examples."""
    project = project.resolve()
    import intent
    import intent.language as language
    from intent.language.builtins import INTRINSICS, Intrinsic, IntrinsicNamespace, QuantFormats
    from intent.language.signatures import INTRINSIC_SIGNATURES

    documents = {}
    paths = [path for directory in (project / "doc/dsl", project / "doc/programming-model")
             for path in directory.glob("*.md")]
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
    section_bodies = {identifier: _section_body(entry["text"])
                      for identifier, entry in documents.items()}
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
                      if ("#L" in d["id"] or d["kind"] != "concept")
                      and pattern.search(section_bodies[d["id"]])]
        symbols[name] = {
            "name": name, "signature": str(signature) if signature else None,
            "declaration": source, "sections": references,
            "canonical": name,
            "members": list(value.members) if isinstance(value, IntrinsicNamespace) else [],
            "availability": "public declaration; backend support and performance are not implied",
        }
    # Core surface shorthands keep their own signatures and share canonical rules.
    for canonical, shorthands in {
        "reduce": ("reduce.sum", "reduce.max", "reduce.any", "reduce.all", "arg_reduce.max"),
        "scan": ("cumsum", "cummax"),
        "contract": ("dot", "matvec", "vecmat", "matmul"),
        "scaled_contract": ("scaled_matmul",),
        "sparse_contract": ("sparse_matmul", "sparse_contract_2to4"),
    }.items():
        rules = [identifier for identifier in symbols[canonical]["sections"]
                 if documents[identifier]["kind"] == "concept"]
        for name in shorthands:
            symbols[name]["canonical"] = canonical
            symbols[name]["sections"] = list(dict.fromkeys((*symbols[name]["sections"], *rules)))
    return {
        "revision": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
        "documents": documents, "symbols": symbols,
    }


class Manual:
    def __init__(self, corpus: dict):
        self.corpus = corpus
        self._section_bodies = {identifier: _section_body(entry["text"])
                                for identifier, entry in corpus["documents"].items()}

    def _rules(self, name: str) -> list[dict]:
        symbol = self.corpus["symbols"][name]
        names = {name.lower(), symbol["canonical"].lower()}
        pattern = re.compile(r"(?<![\w.])(?:i\.)?(?:" +
                             "|".join(re.escape(name) for name in sorted(names)) + r")(?![\w.])")
        exact = re.compile(r"(?<![\w.])(?:i\.)?" + re.escape(name.lower()) + r"(?![\w.])")
        qualified = re.compile(r"(?<![\w.])i\." + re.escape(name.lower()) + r"(?![\w.])")
        canonical = re.compile(r"(?<![\w.])i\." + re.escape(symbol["canonical"].lower()) + r"(?![\w.])")

        def relevance(entry):
            text = self._section_bodies[entry["id"]]
            source = entry["source"]
            if not any(line.strip() and not line.startswith("#") for line in text.splitlines()):
                return (0, 0, entry["id"])
            score = 16 * bool(exact.search(entry["title"].lower()))
            score += 32 * bool(qualified.search(text.lower()))
            score += 16 * bool(canonical.search(text.lower()))
            score += 8 * bool(pattern.search(entry["title"].lower()))
            score += bool(pattern.search(text.lower()))
            score += 20 * (source == "doc/dsl/core.md")
            score += 3 * (source == "doc/dsl/authoring.md")
            return (-score, len(text), entry["id"])

        return sorted((self.corpus["documents"][key] for key in symbol["sections"]
                       if self.corpus["documents"][key]["kind"] == "concept"), key=relevance)

    def search(self, query: str,
               kind: Literal["all", "api", "concept", "diagnostic"] = "all") -> dict:
        """Find API names, language rules and diagnostics. Read returned IDs for full context."""
        if kind not in {"all", "api", "concept", "diagnostic"}:
            raise ValueError("kind must be all, api, concept or diagnostic")
        terms = list(dict.fromkeys(re.findall(r"[\w.]+", query.lower())))
        if not terms:
            raise ValueError("query must contain a name or search term")
        results = []
        if kind in {"all", "api"}:
            api_terms = [term.removeprefix("intent.language.").removeprefix("i.")
                         for term in terms]
            for name, entry in self.corpus["symbols"].items():
                score = sum(16 if term == name.lower() else
                            4 if len(term) > 2 and term in name.lower() else 0
                            for term in api_terms)
                if score:
                    if name.lower() in api_terms:
                        score += 1 / (1 + api_terms.index(name.lower()))
                    results.append((score, {"id": name, "kind": "api", "title": name,
                                                  "signature": entry["signature"]}))
        documents = [entry for entry in self.corpus["documents"].values()
                     if kind in {"all", entry["kind"]}
                     and ("#L" in entry["id"] or entry["kind"] != "concept")]
        patterns = [re.compile(r"(?<![a-z0-9_])" + re.escape(term) +
                               r"(?![a-z0-9_])") if term.isascii()
                    else re.compile(re.escape(term)) for term in terms]
        occurrences = {
            entry["id"]: [(bool(pattern.search(entry["title"].lower())),
                           bool(pattern.search(self._section_bodies[entry["id"]].lower())))
                          for pattern in patterns]
            for entry in documents
        }
        frequencies = [sum(any(matches[index]) for matches in occurrences.values())
                       for index in range(len(terms))]
        weights = [math.log1p(len(documents) / (1 + frequency))
                   for frequency in frequencies]
        for entry in documents:
            score = sum(weight * (8 * title + text)
                        for weight, (title, text) in
                        zip(weights, occurrences[entry["id"]]))
            if kind == "all" and entry["kind"] == "diagnostic":
                score /= 4
            if score:
                match = {key: entry[key] for key in ("id", "title", "kind", "source", "line")}
                lines = self._section_bodies[entry["id"]].splitlines()
                match["excerpt"] = next((line.strip()[:400] for line in lines
                                         if not line.startswith("#") and
                                         any(pattern.search(line.lower()) for pattern in patterns)), "")
                results.append((score, match))
        results.sort(key=lambda row: (-row[0], row[1]["id"]))
        selected = results[:12]
        if kind == "all":
            # Preserve room for language rules when the query also names APIs.
            limits = {"api": 3, "concept": 7, "diagnostic": 2}
            selected = []
            for result in results:
                category = result[1]["kind"]
                if limits[category]:
                    selected.append(result)
                    limits[category] -= 1
            for result in results:
                if len(selected) == 12:
                    break
                if result not in selected:
                    selected.append(result)
            selected.sort(key=lambda row: (-row[0], row[1]["id"]))
        return {"revision": self.corpus["revision"], "matches": [r[1] for r in selected],
                "total": len(results)}

    def api(self, name: str) -> dict:
        """Get an exact declaration and its language rules; read(id=...) includes nested sections."""
        name = name.removeprefix("intent.language.").removeprefix("I.")
        entry = self.corpus["symbols"].get(name)
        if entry is None:
            return {"status": "not-found", "name": name, "message": "Not a current public declaration; no replacement is inferred."}
        rules = self._rules(name)
        return {"status": "declared", "revision": self.corpus["revision"],
                **{key: value for key, value in entry.items() if key != "sections"},
                "signature_note": None if entry["signature"] else "No inspectable signature is declared; consult the linked rules, not a guessed signature.",
                "rules": [{**{field: rule[field]
                              for field in ("id", "title", "source", "line")},
                           "text": self._section_bodies[rule["id"]]}
                          for rule in rules[:4]],
                "additional_rules": [{field: rule[field] for field in ("id", "title", "source", "line")}
                                     for rule in rules[4:]],
                "read_note": "The four closest rules include their own text. Use read on any rule ID for nested sections or additional rules. These contracts do not infer your program's result schema. Implementation diagnostics are available through search(kind='diagnostic').",
                "verification": "not evaluated by this read-only service; diagnostics do not redefine doc semantics"}

    def read(self, id: str, section: str | None = None) -> dict:
        """Read a published document/rule ID or exact API name. section accepts a heading title or its published line (for example L134)."""
        name = id.removeprefix("intent.language.").removeprefix("I.")
        if name in self.corpus["symbols"] and section is None:
            return self.api(name)
        if section is not None:
            anchor = re.fullmatch(r"#?L?(\d+)", section)
            entries = [d for d in self.corpus["documents"].values()
                       if d["source"] == id and "#L" in d["id"] and
                       (d["line"] == int(anchor[1]) if anchor else d["title"] == section)]
            if len(entries) != 1:
                titles = [d["title"] for d in self.corpus["documents"].values()
                          if d["source"] == id and "#L" in d["id"]]
                raise ValueError(f"Use a published heading line or an exact section title from this document: {titles}; "
                                 "omit section to read the whole document, or use api(name=...) for an API.")
            entry = entries[0]
        elif id in self.corpus["documents"]:
            entry = self.corpus["documents"][id]
        else:
            entries = [d for d in self.corpus["documents"].values()
                       if d["source"] == id and "#L" in d["id"]]
            if len(entries) > 1:
                raise ValueError(f"Only these sections are published; read one of their IDs: "
                                 f"{[d['id'] for d in entries]}")
            if not entries:
                raise ValueError("Unknown document/rule ID or API name. Use search(query=...) and copy a returned id; "
                                 "api(name='I.matmul') reads an exact public declaration. Do not invent api/ paths.")
            entry = entries[0]
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
        "read(id=..., section=...) accepts an exact section title or its published heading line. "
        "Use search(query=..., kind=...) to find names and IDs. "
        "No execution or task answers."
    ))
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    for method in (manual.search, manual.api, manual.read):
        server.add_tool(method, annotations=annotations)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
