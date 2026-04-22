"""Rocq source navigation backed by extracted Python source maps."""

import argparse
import json
import re
import sys
from ast import AnnAssign, Assign, ClassDef, FunctionDef, Name, parse
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any
from urllib.parse import unquote, urlparse

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
_KEYWORDS = frozenset(
    {
        "Declare",
        "Definition",
        "Extract",
        "Extraction",
        "Inductive",
        "Lemma",
        "ML",
        "Module",
        "Proof",
        "Qed",
        "Unset",
        "as",
        "else",
        "end",
        "fun",
        "if",
        "in",
        "lambda",
        "match",
        "return",
        "then",
        "with",
    }
)
_SEMANTIC_TOKEN_TYPES = (
    "comment",
    "string",
    "keyword",
    "function",
    "enumMember",
    "type",
)
_SEMANTIC_TOKEN_MODIFIERS = ("declaration", "readonly")


@dataclass(frozen=True)
class Position:
    line: int
    character: int

    def to_lsp(self) -> dict[str, int]:
        return {"line": self.line, "character": self.character}


@dataclass(frozen=True)
class Range:
    start: Position
    end: Position

    def to_lsp(self) -> dict[str, dict[str, int]]:
        return {"start": self.start.to_lsp(), "end": self.end.to_lsp()}


@dataclass(frozen=True)
class Location:
    path: Path
    range: Range

    def to_json(self, repo_root: Path) -> dict[str, Any]:
        return {
            "path": _repo_path(repo_root, self.path),
            "range": self.range.to_lsp(),
        }

    def to_lsp(self) -> dict[str, Any]:
        return {"uri": self.path.resolve().as_uri(), "range": self.range.to_lsp()}


@dataclass(frozen=True)
class Diagnostic:
    message: str
    path: Path | None = None
    range: Range | None = None

    def to_json(self, repo_root: Path) -> dict[str, Any]:
        result: dict[str, Any] = {"message": self.message}
        if self.path is not None:
            result["path"] = _repo_path(repo_root, self.path)
        if self.range is not None:
            result["range"] = self.range.to_lsp()
        return result

    def to_lsp(self) -> dict[str, Any]:
        return {
            "range": (
                self.range
                if self.range is not None
                else Range(Position(0, 0), Position(0, 1))
            ).to_lsp(),
            "severity": 2,
            "source": "fido-rocq-lsp",
            "message": self.message,
        }


@dataclass(frozen=True)
class Symbol:
    name: str
    source: Location
    python: Location | None
    python_signature: str | None

    def hover(self, repo_root: Path) -> dict[str, Any]:
        lines = [
            f"**{self.name}**",
            f"Rocq: `{_repo_path(repo_root, self.source.path)}:{self.source.range.start.line + 1}:{self.source.range.start.character + 1}`",
        ]
        if self.python is not None and self.python_signature is not None:
            lines.append(
                f"Python: `{_repo_path(repo_root, self.python.path)}:{self.python.range.start.line + 1}`"
            )
            lines.append("")
            lines.append("```python")
            lines.append(self.python_signature)
            lines.append("```")
        return {"contents": {"kind": "markdown", "value": "\n".join(lines)}}

    def document_symbol(self, repo_root: Path) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": 12,
            "location": self.source.to_json(repo_root),
        }

    def code_lens(self) -> dict[str, Any] | None:
        if self.python is None:
            return None
        return {
            "range": self.source.range.to_lsp(),
            "command": {
                "title": f"Generated Python: {self.python.path.name}",
                "command": "fido.openGeneratedPython",
                "arguments": [self.python.to_lsp()],
            },
            "data": {
                "symbol": self.name,
                "target": self.python.to_lsp(),
            },
        }


@dataclass(frozen=True)
class SemanticToken:
    line: int
    start: int
    length: int
    token_type: str
    modifiers: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "start": self.start,
            "length": self.length,
            "type": self.token_type,
            "modifiers": list(self.modifiers),
        }


class RocqIndex:
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._models_dir = self._repo_root / "models"
        self._generated_dir = self._repo_root / "src" / "fido" / "rocq"
        self._symbols: dict[str, Symbol] = {}
        self._diagnostics: list[Diagnostic] = []

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return tuple(sorted(self._symbols.values(), key=lambda item: item.name))

    def refresh(self) -> None:
        self._symbols = {}
        self._diagnostics = []
        if not self._generated_dir.is_dir():
            self._diagnostics.append(
                Diagnostic("generated Rocq Python directory is missing")
            )
            return
        for map_path in sorted(self._generated_dir.glob("*.pymap")):
            self._load_map(map_path)

    def symbol_at(self, path: Path, line: int, character: int) -> Symbol | None:
        source = self._resolve(path)
        for symbol in self.symbols_for_file(source):
            if _contains(symbol.source.range, line, character):
                return symbol
        token = self._token_at(source, line, character)
        return self._symbols.get(token) if token else None

    def symbols_for_file(self, path: Path) -> tuple[Symbol, ...]:
        source = self._resolve(path)
        return tuple(symbol for symbol in self.symbols if symbol.source.path == source)

    def semantic_tokens_for_file(self, path: Path) -> tuple[SemanticToken, ...]:
        source = self._resolve(path)
        if not source.is_file():
            return ()
        return _semantic_tokens_for_text(
            source.read_text(), self.symbols_for_file(source)
        )

    def references(self, symbol: Symbol) -> tuple[Location, ...]:
        locations: list[Location] = []
        for path in sorted(self._models_dir.glob("*.v")):
            text = path.read_text()
            for match in _identifier_matches_without_comments(text, symbol.name):
                line, column = _offset_to_position(text, match.start())
                locations.append(
                    Location(
                        path.resolve(),
                        Range(
                            Position(line, column),
                            Position(line, column + len(symbol.name)),
                        ),
                    )
                )
        return tuple(locations)

    def callers(self, symbol: Symbol) -> tuple[Location, ...]:
        return tuple(
            location
            for location in self.references(symbol)
            if location.range != symbol.source.range
            or location.path != symbol.source.path
        )

    def dependency_graph(self) -> dict[str, Any]:
        nodes: dict[str, dict[str, str]] = {}
        edges: list[dict[str, str]] = []
        for symbol in self.symbols:
            source = _repo_path(self._repo_root, symbol.source.path)
            nodes[source] = {"path": source, "kind": "rocq"}
            if symbol.python is None:
                continue
            python = _repo_path(self._repo_root, symbol.python.path)
            map_path = symbol.python.path.with_suffix(".pymap")
            source_map = _repo_path(self._repo_root, map_path)
            nodes[python] = {"path": python, "kind": "python"}
            nodes[source_map] = {"path": source_map, "kind": "sourceMap"}
            edges.append({"from": source, "to": source_map, "kind": "maps"})
            edges.append({"from": source_map, "to": python, "kind": "generates"})
        return {
            "nodes": sorted(nodes.values(), key=lambda item: item["path"]),
            "edges": sorted(edges, key=lambda item: (item["from"], item["to"])),
        }

    def diagnostics_for_file(self, path: Path | None = None) -> tuple[Diagnostic, ...]:
        if path is None:
            return self.diagnostics
        resolved = self._resolve(path)
        return tuple(
            diag
            for diag in self._diagnostics
            if diag.path is None or diag.path.resolve() == resolved
        )

    def _load_map(self, map_path: Path) -> None:
        try:
            raw = json.loads(map_path.read_text())
        except json.JSONDecodeError as exc:
            self._diagnostics.append(
                Diagnostic(f"bad source map {map_path.name}: {exc}")
            )
            return
        if raw.get("version") != 1:
            self._diagnostics.append(
                Diagnostic(f"unsupported source map version in {map_path.name}")
            )
            return
        python_path = self._generated_dir / str(raw.get("python_file", ""))
        signatures = _python_signatures(python_path)
        for entry in raw.get("entries", []):
            if isinstance(entry, dict):
                self._load_entry(map_path, python_path, signatures, entry)

    def _load_entry(
        self,
        map_path: Path,
        python_path: Path,
        signatures: dict[str, tuple[str, Range]],
        entry: dict[str, Any],
    ) -> None:
        name = str(entry.get("symbol", ""))
        if not name:
            return
        source = self._source_path(str(entry.get("source_file", "")))
        if source is None:
            self._diagnostics.append(
                Diagnostic(f"{map_path.name}: source file missing for {name}")
            )
            return
        start_line = int(entry["source_start_line"]) - 1
        start_col = int(entry["source_start_col"])
        source_location = Location(
            source,
            Range(
                Position(start_line, start_col),
                Position(start_line, start_col + len(name)),
            ),
        )
        python_info = signatures.get(name)
        python_location: Location | None = None
        python_signature: str | None = None
        if python_info is not None:
            python_signature, python_range = python_info
            python_location = Location(python_path.resolve(), python_range)
        else:
            self._diagnostics.append(
                Diagnostic(f"generated Python declaration missing for {name}", source)
            )
        if name in self._symbols:
            self._diagnostics.append(Diagnostic(f"duplicate extracted symbol: {name}"))
        self._symbols[name] = Symbol(
            name=name,
            source=source_location,
            python=python_location,
            python_signature=python_signature,
        )
        if (
            python_path.is_file()
            and source.stat().st_mtime > python_path.stat().st_mtime
        ):
            self._diagnostics.append(
                Diagnostic(f"generated Python may be stale for {name}", source)
            )

    def _source_path(self, name: str) -> Path | None:
        for path in self._models_dir.glob("*.v"):
            if path.name == Path(name).name:
                return path.resolve()
        return None

    def _resolve(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else self._repo_root / path
        return candidate.resolve()

    def _token_at(self, path: Path, line: int, character: int) -> str | None:
        if not path.is_file():
            return None
        lines = path.read_text().splitlines()
        if line < 0 or line >= len(lines):
            return None
        row = lines[line]
        if character < 0 or character > len(row):
            return None
        for match in _IDENT_RE.finditer(row):
            if match.start() <= character < match.end():
                return match.group(0)
        return None


class RocqLanguageService:
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._index = RocqIndex(repo_root)

    def hover(self, path: Path, line: int, character: int) -> dict[str, Any] | None:
        symbol = self._fresh().symbol_at(path, line, character)
        return None if symbol is None else symbol.hover(self._repo_root)

    def definition(self, path: Path, line: int, character: int) -> list[dict[str, Any]]:
        symbol = self._fresh().symbol_at(path, line, character)
        if symbol is None:
            return []
        locations = [symbol.source]
        if symbol.python is not None:
            locations.append(symbol.python)
        return [location.to_json(self._repo_root) for location in locations]

    def references(self, path: Path, line: int, character: int) -> list[dict[str, Any]]:
        index = self._fresh()
        symbol = index.symbol_at(path, line, character)
        if symbol is None:
            return []
        return [
            location.to_json(self._repo_root) for location in index.references(symbol)
        ]

    def callers(self, path: Path, line: int, character: int) -> list[dict[str, Any]]:
        index = self._fresh()
        symbol = index.symbol_at(path, line, character)
        if symbol is None:
            return []
        return [location.to_json(self._repo_root) for location in index.callers(symbol)]

    def symbols(self, path: Path | None = None) -> list[dict[str, Any]]:
        index = self._fresh()
        symbols = index.symbols if path is None else index.symbols_for_file(path)
        return [symbol.document_symbol(self._repo_root) for symbol in symbols]

    def semantic_tokens(self, path: Path) -> list[dict[str, Any]]:
        return [
            token.to_json() for token in self._fresh().semantic_tokens_for_file(path)
        ]

    def code_lens(self, path: Path) -> list[dict[str, Any]]:
        lenses: list[dict[str, Any]] = []
        for symbol in self._fresh().symbols_for_file(path):
            lens = symbol.code_lens()
            if lens is not None:
                lenses.append(lens)
        return lenses

    def diagnostics(self, path: Path | None = None) -> list[dict[str, Any]]:
        return [
            diag.to_json(self._repo_root)
            for diag in self._fresh().diagnostics_for_file(path)
        ]

    def dependency_graph(self) -> dict[str, Any]:
        return self._fresh().dependency_graph()

    def signature_help(
        self, path: Path, line: int, character: int
    ) -> dict[str, Any] | None:
        symbol = self._fresh().symbol_at(path, line, character)
        if symbol is None or symbol.python_signature is None:
            return None
        return {
            "signatures": [
                {
                    "label": symbol.python_signature,
                    "documentation": {
                        "kind": "markdown",
                        "value": f"Extracted from `{_repo_path(self._repo_root, symbol.source.path)}`.",
                    },
                }
            ],
            "activeSignature": 0,
            "activeParameter": 0,
        }

    def completion(self, path: Path, line: int, character: int) -> list[dict[str, Any]]:
        prefix = self._completion_prefix(path, line, character)
        return [
            {
                "label": symbol.name,
                "kind": 3 if _symbol_token_type(symbol.name) == "function" else 20,
                "detail": symbol.python_signature,
                "data": {"source": _repo_path(self._repo_root, symbol.source.path)},
            }
            for symbol in self._fresh().symbols
            if symbol.name.startswith(prefix)
        ]

    def explain(self, path: Path, line: int, character: int) -> dict[str, Any] | None:
        index = self._fresh()
        symbol = index.symbol_at(path, line, character)
        if symbol is None:
            return None
        return {
            "symbol": symbol.name,
            "rocq": symbol.source.to_json(self._repo_root),
            "python": None
            if symbol.python is None
            else symbol.python.to_json(self._repo_root),
            "signature": symbol.python_signature,
            "references": len(index.references(symbol)),
            "diagnostics": [
                diag.to_json(self._repo_root)
                for diag in index.diagnostics_for_file(symbol.source.path)
            ],
        }

    def code_actions(self, path: Path) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = [
            {
                "title": "Refresh Rocq extraction",
                "kind": "source.fixAll.fido.rocq",
                "command": {"command": "fido.makeRocq", "title": "./fido make-rocq"},
            }
        ]
        for symbol in self._fresh().symbols_for_file(path):
            if symbol.python is not None:
                actions.append(
                    {
                        "title": f"Open generated Python for {symbol.name}",
                        "kind": "quickfix",
                        "command": {
                            "command": "fido.openGeneratedPython",
                            "title": f"Open {symbol.python.path.name}",
                            "arguments": [symbol.python.to_lsp()],
                        },
                    }
                )
        return actions

    def prepare_rename(
        self, path: Path, line: int, character: int
    ) -> dict[str, Any] | None:
        symbol = self._fresh().symbol_at(path, line, character)
        return None if symbol is None else symbol.source.range.to_lsp()

    def rename(
        self, path: Path, line: int, character: int, new_name: str
    ) -> dict[str, Any]:
        index = self._fresh()
        symbol = index.symbol_at(path, line, character)
        if symbol is None:
            return {"changes": {}}
        changes: dict[str, list[dict[str, Any]]] = {}
        for location in index.references(symbol):
            uri = location.path.resolve().as_uri()
            changes.setdefault(uri, []).append(
                {"range": location.range.to_lsp(), "newText": new_name}
            )
        return {"changes": changes}

    def lsp_definition(
        self, path: Path, line: int, character: int
    ) -> list[dict[str, Any]]:
        symbol = self._fresh().symbol_at(path, line, character)
        if symbol is None:
            return []
        locations = [symbol.source]
        if symbol.python is not None:
            locations.append(symbol.python)
        return [location.to_lsp() for location in locations]

    def lsp_references(
        self, path: Path, line: int, character: int
    ) -> list[dict[str, Any]]:
        index = self._fresh()
        symbol = index.symbol_at(path, line, character)
        if symbol is None:
            return []
        return [location.to_lsp() for location in index.references(symbol)]

    def lsp_signature_help(
        self, path: Path, line: int, character: int
    ) -> dict[str, Any] | None:
        return self.signature_help(path, line, character)

    def lsp_completion(
        self, path: Path, line: int, character: int
    ) -> list[dict[str, Any]]:
        return self.completion(path, line, character)

    def lsp_document_symbols(self, path: Path) -> list[dict[str, Any]]:
        return [
            {
                "name": symbol.name,
                "kind": 12,
                "range": symbol.source.range.to_lsp(),
                "selectionRange": symbol.source.range.to_lsp(),
            }
            for symbol in self._fresh().symbols_for_file(path)
        ]

    def lsp_semantic_tokens_full(self, path: Path) -> dict[str, list[int]]:
        return {
            "data": _encode_semantic_tokens(
                self._fresh().semantic_tokens_for_file(path)
            )
        }

    def lsp_code_lens(self, path: Path) -> list[dict[str, Any]]:
        return self.code_lens(path)

    def lsp_code_actions(self, path: Path) -> list[dict[str, Any]]:
        return self.code_actions(path)

    def lsp_workspace_symbols(self, query: str) -> list[dict[str, Any]]:
        query_folded = query.lower()
        return [
            {
                "name": symbol.name,
                "kind": 12,
                "location": symbol.source.to_lsp(),
            }
            for symbol in self._fresh().symbols
            if query_folded in symbol.name.lower()
        ]

    def lsp_diagnostics(self, path: Path) -> dict[str, Any]:
        diagnostics = self._fresh().diagnostics_for_file(path)
        return {"kind": "full", "items": [diag.to_lsp() for diag in diagnostics]}

    def _completion_prefix(self, path: Path, line: int, character: int) -> str:
        resolved = self._repo_root / path if not path.is_absolute() else path
        if not resolved.is_file():
            return ""
        lines = resolved.read_text().splitlines()
        if line < 0 or line >= len(lines):
            return ""
        row = lines[line]
        start = min(max(character, 0), len(row))
        while start > 0 and _IDENT_RE.match(row[start - 1]):
            start -= 1
        return row[start:character]

    def _fresh(self) -> RocqIndex:
        self._index.refresh()
        return self._index


class RocqLspServer:
    def __init__(
        self,
        service: RocqLanguageService,
        stdin: IO[str],
        stdout: IO[str],
    ) -> None:
        self._service = service
        self._stdin = stdin
        self._stdout = stdout
        self._running = True

    def run(self) -> int:
        while self._running:
            message = self._read_message()
            if message is None:
                return 0
            self._handle(message)
        return 0

    def _handle(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        try:
            result = self._dispatch(str(method), message.get("params") or {})
        except Exception as exc:
            if request_id is not None:
                self._write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": str(exc)},
                    }
                )
            return
        if request_id is not None:
            self._write({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "capabilities": {
                    "hoverProvider": True,
                    "definitionProvider": True,
                    "referencesProvider": True,
                    "renameProvider": {"prepareProvider": True},
                    "completionProvider": {"triggerCharacters": [".", "_"]},
                    "signatureHelpProvider": {"triggerCharacters": [" ", "("]},
                    "codeActionProvider": True,
                    "documentSymbolProvider": True,
                    "workspaceSymbolProvider": True,
                    "codeLensProvider": {"resolveProvider": False},
                    "semanticTokensProvider": {
                        "legend": {
                            "tokenTypes": list(_SEMANTIC_TOKEN_TYPES),
                            "tokenModifiers": list(_SEMANTIC_TOKEN_MODIFIERS),
                        },
                        "full": True,
                    },
                    "diagnosticProvider": {
                        "interFileDependencies": True,
                        "workspaceDiagnostics": False,
                    },
                }
            }
        if method == "shutdown":
            self._running = False
            return None
        if method == "exit":
            self._running = False
            return None
        if method == "textDocument/hover":
            path, line, character = _lsp_text_position(params)
            return self._service.hover(path, line, character)
        if method == "textDocument/definition":
            path, line, character = _lsp_text_position(params)
            return self._service.lsp_definition(path, line, character)
        if method == "textDocument/references":
            path, line, character = _lsp_text_position(params)
            return self._service.lsp_references(path, line, character)
        if method == "textDocument/signatureHelp":
            path, line, character = _lsp_text_position(params)
            return self._service.lsp_signature_help(path, line, character)
        if method == "textDocument/completion":
            path, line, character = _lsp_text_position(params)
            return self._service.lsp_completion(path, line, character)
        if method == "textDocument/documentSymbol":
            return self._service.lsp_document_symbols(
                _path_from_uri(params["textDocument"]["uri"])
            )
        if method == "textDocument/codeAction":
            return self._service.lsp_code_actions(
                _path_from_uri(params["textDocument"]["uri"])
            )
        if method == "textDocument/codeLens":
            return self._service.lsp_code_lens(
                _path_from_uri(params["textDocument"]["uri"])
            )
        if method == "textDocument/prepareRename":
            path, line, character = _lsp_text_position(params)
            return self._service.prepare_rename(path, line, character)
        if method == "textDocument/rename":
            path, line, character = _lsp_text_position(params)
            return self._service.rename(path, line, character, str(params["newName"]))
        if method == "textDocument/semanticTokens/full":
            return self._service.lsp_semantic_tokens_full(
                _path_from_uri(params["textDocument"]["uri"])
            )
        if method == "workspace/symbol":
            return self._service.lsp_workspace_symbols(str(params.get("query", "")))
        if method == "textDocument/diagnostic":
            return self._service.lsp_diagnostics(
                _path_from_uri(params["textDocument"]["uri"])
            )
        return None

    def _read_message(self) -> dict[str, Any] | None:
        length: int | None = None
        while True:
            line = self._stdin.readline()
            if line == "":
                return None
            if line in ("\r\n", "\n"):
                break
            name, _, value = line.partition(":")
            if name.lower() == "content-length":
                length = int(value.strip())
        if length is None:
            return None
        body = self._stdin.read(length)
        return json.loads(body)

    def _write(self, message: dict[str, Any]) -> None:
        body = json.dumps(message, separators=(",", ":"))
        self._stdout.write(f"Content-Length: {len(body.encode())}\r\n\r\n{body}")
        self._stdout.flush()


class RocqLspCli:
    def __init__(
        self,
        repo_root: Path,
        stdout: IO[str],
        stderr: IO[str],
    ) -> None:
        self._repo_root = repo_root
        self._stdout = stdout
        self._stderr = stderr

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="lsp")
        subcommands = parser.add_subparsers(dest="command", required=True)
        for command in (
            "hover",
            "definition",
            "references",
            "callers",
            "signature",
            "completion",
            "explain",
        ):
            sub = subcommands.add_parser(command)
            sub.add_argument("file")
            sub.add_argument("--line", required=True, type=int)
            sub.add_argument("--column", required=True, type=int)
            sub.add_argument("--json", action="store_true", required=True)
        symbols = subcommands.add_parser("symbols")
        symbols.add_argument("file", nargs="?")
        symbols.add_argument("--workspace", action="store_true")
        symbols.add_argument("--json", action="store_true", required=True)
        tokens = subcommands.add_parser("tokens")
        tokens.add_argument("file")
        tokens.add_argument("--json", action="store_true", required=True)
        codelens = subcommands.add_parser("codelens")
        codelens.add_argument("file")
        codelens.add_argument("--json", action="store_true", required=True)
        codeactions = subcommands.add_parser("codeactions")
        codeactions.add_argument("file")
        codeactions.add_argument("--json", action="store_true", required=True)
        graph = subcommands.add_parser("graph")
        graph.add_argument("--json", action="store_true", required=True)
        rename = subcommands.add_parser("rename")
        rename.add_argument("file")
        rename.add_argument("--line", required=True, type=int)
        rename.add_argument("--column", required=True, type=int)
        rename.add_argument("--new-name", required=True)
        rename.add_argument("--json", action="store_true", required=True)
        diagnostics = subcommands.add_parser("diagnostics")
        diagnostics.add_argument("file", nargs="?")
        diagnostics.add_argument("--json", action="store_true", required=True)
        args = parser.parse_args(argv)

        service = RocqLanguageService(self._repo_root)
        try:
            result = self._run_command(service, args)
        except OSError as exc:
            print(f"error: {exc}", file=self._stderr)
            return 1
        self._stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0

    def _run_command(
        self, service: RocqLanguageService, args: argparse.Namespace
    ) -> Any:
        if args.command == "hover":
            return service.hover(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "definition":
            return service.definition(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "references":
            return service.references(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "callers":
            return service.callers(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "signature":
            return service.signature_help(
                Path(args.file), args.line - 1, args.column - 1
            )
        if args.command == "completion":
            return service.completion(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "explain":
            return service.explain(Path(args.file), args.line - 1, args.column - 1)
        if args.command == "symbols":
            return service.symbols(None if args.workspace else Path(args.file))
        if args.command == "tokens":
            return service.semantic_tokens(Path(args.file))
        if args.command == "codelens":
            return service.code_lens(Path(args.file))
        if args.command == "codeactions":
            return service.code_actions(Path(args.file))
        if args.command == "graph":
            return service.dependency_graph()
        if args.command == "rename":
            return service.rename(
                Path(args.file), args.line - 1, args.column - 1, str(args.new_name)
            )
        if args.command == "diagnostics":
            return service.diagnostics(Path(args.file) if args.file else None)
        raise ValueError(f"unsupported command: {args.command}")


def _repo_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _contains(value: Range, line: int, character: int) -> bool:
    if line != value.start.line:
        return False
    return value.start.character <= character <= value.end.character


def _python_signatures(path: Path) -> dict[str, tuple[str, Range]]:
    if not path.is_file():
        return {}
    text = path.read_text()
    tree = parse(text)
    lines = text.splitlines()
    signatures: dict[str, tuple[str, Range]] = {}
    for node in tree.body:
        if isinstance(node, FunctionDef | ClassDef):
            start = node.lineno - 1
            end = _signature_end_line(lines, start)
            signature = " ".join(
                _strip_generated_comment(line).strip()
                for line in lines[start : end + 1]
            )
            if signature.endswith(":"):
                signature = signature[:-1]
            signatures[node.name] = (
                signature,
                Range(Position(start, 0), Position(end, len(lines[end]))),
            )
        elif isinstance(node, AnnAssign) and isinstance(node.target, Name):
            name = node.target.id
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            signature = " ".join(
                _strip_generated_comment(line).strip()
                for line in lines[start : end + 1]
            )
            signatures[name] = (
                signature,
                Range(Position(start, 0), Position(end, len(lines[end]))),
            )
        elif isinstance(node, Assign):
            target = next(
                (target for target in node.targets if isinstance(target, Name)),
                None,
            )
            if target is None:
                continue
            start = node.lineno - 1
            end = (node.end_lineno or node.lineno) - 1
            signature = " ".join(
                _strip_generated_comment(line).strip()
                for line in lines[start : end + 1]
            )
            signatures[target.id] = (
                signature,
                Range(Position(start, 0), Position(end, len(lines[end]))),
            )
    return signatures


def _signature_end_line(lines: list[str], start: int) -> int:
    balance = 0
    for index in range(start, len(lines)):
        line = lines[index]
        balance += line.count("(") - line.count(")")
        if balance <= 0 and line.rstrip().endswith(":"):
            return index
    return start


def _strip_generated_comment(line: str) -> str:
    return line.split("  # From ", 1)[0].rstrip()


def _identifier_matches_without_comments(
    text: str, name: str
) -> tuple[re.Match[str], ...]:
    masked = _mask_comments_and_strings(text)
    return tuple(
        match for match in _IDENT_RE.finditer(masked) if match.group(0) == name
    )


def _semantic_tokens_for_text(
    text: str, symbols: tuple[Symbol, ...]
) -> tuple[SemanticToken, ...]:
    tokens = [
        *[
            SemanticToken(line, start, length, kind)
            for line, start, length, kind in _comment_and_string_ranges(text)
        ],
        *[
            SemanticToken(line, start, len(word), "keyword")
            for line, start, word in _keyword_ranges(text)
        ],
        *[
            SemanticToken(
                symbol.source.range.start.line,
                symbol.source.range.start.character,
                len(symbol.name),
                _symbol_token_type(symbol.name),
                ("declaration", "readonly"),
            )
            for symbol in symbols
        ],
    ]
    return _dedupe_semantic_tokens(tokens)


def _comment_and_string_ranges(text: str) -> tuple[tuple[int, int, int, str], ...]:
    ranges: list[tuple[int, int, int, str]] = []
    index = 0
    comment_depth = 0
    comment_start: int | None = None
    string_start: int | None = None
    in_string = False
    while index < len(text):
        pair = text[index : index + 2]
        char = text[index]
        if comment_depth:
            if pair == "(*":
                comment_depth += 1
                index += 2
                continue
            if pair == "*)":
                comment_depth -= 1
                index += 2
                if comment_depth == 0 and comment_start is not None:
                    ranges.extend(
                        _split_range_by_line(text, comment_start, index, "comment")
                    )
                    comment_start = None
                continue
            index += 1
            continue
        if in_string:
            if char == "\\":
                index += 2
                continue
            if char == '"':
                index += 1
                if string_start is not None:
                    ranges.extend(
                        _split_range_by_line(text, string_start, index, "string")
                    )
                string_start = None
                in_string = False
                continue
            index += 1
            continue
        if pair == "(*":
            comment_start = index
            comment_depth = 1
            index += 2
        elif char == '"':
            string_start = index
            in_string = True
            index += 1
        else:
            index += 1
    if comment_depth and comment_start is not None:
        ranges.extend(_split_range_by_line(text, comment_start, len(text), "comment"))
    if in_string and string_start is not None:
        ranges.extend(_split_range_by_line(text, string_start, len(text), "string"))
    return tuple(ranges)


def _keyword_ranges(text: str) -> tuple[tuple[int, int, str], ...]:
    masked = _mask_comments_and_strings(text)
    ranges: list[tuple[int, int, str]] = []
    for match in _IDENT_RE.finditer(masked):
        word = match.group(0)
        if word in _KEYWORDS:
            line, column = _offset_to_position(masked, match.start())
            ranges.append((line, column, word))
    return tuple(ranges)


def _split_range_by_line(
    text: str, start: int, end: int, kind: str
) -> tuple[tuple[int, int, int, str], ...]:
    ranges: list[tuple[int, int, int, str]] = []
    offset = start
    while offset < end:
        line, column = _offset_to_position(text, offset)
        next_newline = text.find("\n", offset, end)
        line_end = end if next_newline == -1 else next_newline
        length = line_end - offset
        if length > 0:
            ranges.append((line, column, length, kind))
        offset = line_end + 1
    return tuple(ranges)


def _symbol_token_type(name: str) -> str:
    return "enumMember" if name[:1].isupper() else "function"


def _dedupe_semantic_tokens(
    tokens: list[SemanticToken],
) -> tuple[SemanticToken, ...]:
    seen: set[tuple[int, int, int]] = set()
    result: list[SemanticToken] = []
    for token in sorted(tokens, key=lambda item: (item.line, item.start, item.length)):
        key = (token.line, token.start, token.length)
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return tuple(result)


def _encode_semantic_tokens(tokens: tuple[SemanticToken, ...]) -> list[int]:
    data: list[int] = []
    previous_line = 0
    previous_start = 0
    for token in tokens:
        delta_line = token.line - previous_line
        delta_start = token.start if delta_line else token.start - previous_start
        data.extend(
            [
                delta_line,
                delta_start,
                token.length,
                _SEMANTIC_TOKEN_TYPES.index(token.token_type),
                _semantic_modifier_bits(token.modifiers),
            ]
        )
        previous_line = token.line
        previous_start = token.start
    return data


def _semantic_modifier_bits(modifiers: tuple[str, ...]) -> int:
    bits = 0
    for modifier in modifiers:
        bits |= 1 << _SEMANTIC_TOKEN_MODIFIERS.index(modifier)
    return bits


def _mask_comments_and_strings(text: str) -> str:
    chars = list(text)
    index = 0
    comment_depth = 0
    in_string = False
    while index < len(chars):
        pair = text[index : index + 2]
        char = text[index]
        if comment_depth:
            chars[index] = "\n" if char == "\n" else " "
            if pair == "(*":
                chars[index + 1] = " "
                comment_depth += 1
                index += 2
            elif pair == "*)":
                chars[index + 1] = " "
                comment_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if in_string:
            chars[index] = " "
            if char == "\\":
                if index + 1 < len(chars):
                    chars[index + 1] = " "
                index += 2
            elif char == '"':
                in_string = False
                index += 1
            else:
                index += 1
            continue
        if pair == "(*":
            chars[index] = " "
            chars[index + 1] = " "
            comment_depth = 1
            index += 2
        elif char == '"':
            chars[index] = " "
            in_string = True
            index += 1
        else:
            index += 1
    return "".join(chars)


def _offset_to_position(text: str, offset: int) -> tuple[int, int]:
    prefix = text[:offset]
    line = prefix.count("\n")
    last_newline = prefix.rfind("\n")
    column = offset if last_newline == -1 else offset - last_newline - 1
    return line, column


def _path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _lsp_text_position(params: dict[str, Any]) -> tuple[Path, int, int]:
    path = _path_from_uri(params["textDocument"]["uri"])
    position = params["position"]
    return path, int(position["line"]), int(position["character"])


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main_cli() -> int:
    return RocqLspCli(repo_root(), sys.stdout, sys.stderr).run(sys.argv[1:])


def main_lsp() -> int:
    return RocqLspServer(RocqLanguageService(repo_root()), sys.stdin, sys.stdout).run()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        return RocqLspServer(
            RocqLanguageService(repo_root()), sys.stdin, sys.stdout
        ).run()
    return RocqLspCli(repo_root(), sys.stdout, sys.stderr).run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
