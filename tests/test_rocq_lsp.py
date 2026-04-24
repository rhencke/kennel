# pyright: reportPrivateUsage=false

import json
import os
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from fido import rocq_lsp
from fido.rocq_lsp import (
    Diagnostic,
    Location,
    Position,
    Range,
    RocqIndex,
    RocqLanguageService,
    RocqLspCli,
    RocqLspServer,
)

REPO = Path(__file__).resolve().parents[1]


def _line_containing(path: Path, fragment: str) -> int:
    for index, line in enumerate(path.read_text().splitlines()):
        if fragment in line:
            return index
    raise AssertionError(f"{fragment!r} not found in {path}")


def test_location_and_diagnostic_json_shapes() -> None:
    location = Location(
        REPO / "models" / "session_lock.v",
        Range(Position(66, 11), Position(66, 21)),
    )
    diagnostic = Diagnostic("problem", location.path, location.range)

    assert location.to_json(REPO) == {
        "path": "models/session_lock.v",
        "range": {
            "start": {"line": 66, "character": 11},
            "end": {"line": 66, "character": 21},
        },
    }
    assert location.to_lsp()["uri"].startswith("file://")
    assert diagnostic.to_json(REPO)["message"] == "problem"
    assert diagnostic.to_lsp()["source"] == "fido-rocq-lsp"
    assert Diagnostic("global").to_json(REPO) == {"message": "global"}


def test_index_finds_transition_symbol_and_python_signature() -> None:
    index = RocqIndex(REPO)
    index.refresh()

    symbol = index.symbol_at(REPO / "models" / "session_lock.v", 66, 11)

    assert symbol is not None
    assert symbol.name == "transition"
    assert symbol.source.path == (REPO / "models" / "session_lock.v").resolve()
    assert symbol.python is not None
    assert symbol.python.path.name == "transition.py"
    assert symbol.python.range.start.line > 0
    assert symbol.python.range.start.character == 0
    assert symbol.python.range.end.character > len("transition")
    assert (
        symbol.python_signature
        == "def transition(current: State, event0: Event) -> State | None"
    )
    assert all(diag.message for diag in index.diagnostics)


def test_service_hover_definition_references_symbols_and_diagnostics() -> None:
    service = RocqLanguageService(REPO)

    hover = service.hover(Path("models/session_lock.v"), 66, 11)
    definitions = service.definition(Path("models/session_lock.v"), 66, 11)
    references = service.references(Path("models/session_lock.v"), 66, 11)
    callers = service.callers(Path("models/session_lock.v"), 66, 11)
    symbols = service.symbols(Path("models/session_lock.v"))
    tokens = service.semantic_tokens(Path("models/session_lock.v"))
    lenses = service.code_lens(Path("models/session_lock.v"))
    graph = service.dependency_graph()
    signature = service.signature_help(Path("models/session_lock.v"), 66, 11)
    completions = service.completion(Path("models/session_lock.v"), 66, 14)
    explanation = service.explain(Path("models/session_lock.v"), 66, 11)
    code_actions = service.code_actions(Path("models/session_lock.v"))
    rename = service.rename(Path("models/session_lock.v"), 66, 11, "step")

    assert hover is not None
    assert (
        "def transition(current: State, event0: Event) -> State | None"
        in hover["contents"]["value"]
    )
    assert [item["path"] for item in definitions] == [
        "models/session_lock.v",
        "src/fido/rocq/transition.py",
    ]
    assert any(item["path"] == "models/session_lock.v" for item in references)
    assert callers
    assert symbols == [
        {
            "name": "transition",
            "kind": 12,
            "location": definitions[0],
        }
    ]
    assert {
        "line": 66,
        "start": 11,
        "length": 10,
        "type": "function",
        "modifiers": ["declaration", "readonly"],
    } in tokens
    assert lenses[0]["command"]["title"] == "Generated Python: transition.py"
    assert lenses[0]["data"]["symbol"] == "transition"
    assert any(node["path"] == "models/session_lock.v" for node in graph["nodes"])
    assert any(edge["kind"] == "generates" for edge in graph["edges"])
    assert signature is not None
    assert signature["signatures"][0]["label"].startswith("def transition")
    assert completions[0]["label"] == "transition"
    assert explanation is not None
    assert explanation["references"] == len(references)
    assert code_actions[0]["command"]["command"] == "fido.makeRocq"
    assert rename["changes"]
    assert service.symbols(None)
    assert all("message" in item for item in service.diagnostics())
    assert service.hover(Path("models/session_lock.v"), 0, 0) is None
    assert service.definition(Path("models/session_lock.v"), 0, 0) == []
    assert service.references(Path("models/session_lock.v"), 0, 0) == []
    assert service.callers(Path("models/session_lock.v"), 0, 0) == []
    assert service.signature_help(Path("models/session_lock.v"), 0, 0) is None
    assert service._completion_prefix(Path("missing.v"), 0, 0) == ""
    assert service._completion_prefix(Path("models/session_lock.v"), -1, 0) == ""
    assert service.explain(Path("models/session_lock.v"), 0, 0) is None
    assert service.rename(Path("models/session_lock.v"), 0, 0, "missing") == {
        "changes": {}
    }


def test_lsp_shapes() -> None:
    service = RocqLanguageService(REPO)
    uri = (REPO / "models" / "session_lock.v").resolve().as_uri()

    definitions = service.lsp_definition(Path("models/session_lock.v"), 66, 11)
    references = service.lsp_references(Path("models/session_lock.v"), 66, 11)
    document_symbols = service.lsp_document_symbols(Path("models/session_lock.v"))
    workspace_symbols = service.lsp_workspace_symbols("trans")
    diagnostics = service.lsp_diagnostics(Path("models/session_lock.v"))
    semantic_tokens = service.lsp_semantic_tokens_full(Path("models/session_lock.v"))
    code_lens = service.lsp_code_lens(Path("models/session_lock.v"))
    signature = service.lsp_signature_help(Path("models/session_lock.v"), 66, 11)
    completions = service.lsp_completion(Path("models/session_lock.v"), 66, 14)
    actions = service.lsp_code_actions(Path("models/session_lock.v"))

    assert definitions[0]["uri"] == uri
    assert references
    assert document_symbols[0]["name"] == "transition"
    assert workspace_symbols[0]["location"]["uri"] == uri
    assert diagnostics == {"kind": "full", "items": []}
    assert len(semantic_tokens["data"]) % 5 == 0
    assert code_lens[0]["data"]["target"]["uri"].endswith("/transition.py")
    assert signature is not None
    assert completions[0]["label"] == "transition"
    assert actions[0]["title"] == "Refresh Rocq extraction"
    assert service.lsp_definition(Path("models/session_lock.v"), 0, 0) == []
    assert service.lsp_references(Path("models/session_lock.v"), 0, 0) == []


def test_index_resolves_generated_python_methods_back_to_rocq_symbols() -> None:
    index = RocqIndex(REPO)
    index.refresh()

    coord_index_path = REPO / "src" / "fido" / "rocq" / "coord_index.py"
    add_claim_line = _line_containing(coord_index_path, "    def add_claim(")

    symbol = index.symbol_at(coord_index_path, add_claim_line, 8)

    assert symbol is not None
    assert symbol.name == "coord_add_claim"
    assert symbol.python is not None
    assert symbol.python.path == coord_index_path.resolve()
    assert symbol.python_signature == "def add_claim(self, thread: int) -> CoordIndex"


def test_index_maps_runtime_marker_symbols_to_runtime_python_methods() -> None:
    index = RocqIndex(REPO)
    index.refresh()

    runtime_path = REPO / "src" / "fido" / "rocq" / "concurrency_primitives.py"
    pure_line = _line_containing(runtime_path, "def pure(cls, value:")

    symbol = index.symbol_at(runtime_path, pure_line, 8)

    assert symbol is not None
    assert symbol.name == "io_pure"
    assert symbol.python is not None
    assert symbol.python.path == runtime_path.resolve()
    assert symbol.python_signature.startswith("def pure(cls, value:")


def test_cli_outputs_json_for_each_command() -> None:
    for argv in (
        ["hover", "models/session_lock.v", "--line", "67", "--column", "12", "--json"],
        [
            "definition",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--json",
        ],
        [
            "references",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--json",
        ],
        [
            "callers",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--json",
        ],
        [
            "signature",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--json",
        ],
        [
            "completion",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "15",
            "--json",
        ],
        [
            "explain",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--json",
        ],
        ["symbols", "models/session_lock.v", "--json"],
        ["symbols", "--workspace", "--json"],
        ["tokens", "models/session_lock.v", "--json"],
        ["codelens", "models/session_lock.v", "--json"],
        ["codeactions", "models/session_lock.v", "--json"],
        ["graph", "--json"],
        [
            "rename",
            "models/session_lock.v",
            "--line",
            "67",
            "--column",
            "12",
            "--new-name",
            "step",
            "--json",
        ],
        ["diagnostics", "models/session_lock.v", "--json"],
        ["diagnostics", "--json"],
    ):
        out = StringIO()
        result = RocqLspCli(REPO, out, StringIO()).run(argv)
        assert result == 0
        assert json.loads(out.getvalue()) is not None


def test_cli_reports_os_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_self: RocqLspCli, _service: RocqLanguageService, _args: Any) -> Any:
        raise OSError("nope")

    monkeypatch.setattr(RocqLspCli, "_run_command", boom)
    err = StringIO()

    result = RocqLspCli(REPO, StringIO(), err).run(
        ["diagnostics", "missing.v", "--json"]
    )

    assert result == 1
    assert "nope" in err.getvalue()


def test_lsp_server_handles_requests_notifications_errors_and_eof() -> None:
    uri = (REPO / "models" / "session_lock.v").resolve().as_uri()
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "initialized", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "textDocument/references",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "textDocument/documentSymbol",
            "params": {"textDocument": {"uri": uri}},
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "workspace/symbol",
            "params": {"query": "transition"},
        },
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "textDocument/signatureHelp",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 14},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "textDocument/codeAction",
            "params": {"textDocument": {"uri": uri}},
        },
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "textDocument/codeLens",
            "params": {"textDocument": {"uri": uri}},
        },
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "textDocument/semanticTokens/full",
            "params": {"textDocument": {"uri": uri}},
        },
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "textDocument/prepareRename",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "textDocument/rename",
            "params": {
                "textDocument": {"uri": uri},
                "position": {"line": 66, "character": 11},
                "newName": "step",
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 14,
            "method": "textDocument/diagnostic",
            "params": {"textDocument": {"uri": uri}},
        },
        {"jsonrpc": "2.0", "id": 15, "method": "unknown", "params": {}},
        {"jsonrpc": "2.0", "id": 16, "method": "textDocument/hover", "params": {}},
        {"jsonrpc": "2.0", "id": 17, "method": "shutdown", "params": {}},
    ]
    stdout = StringIO()

    result = RocqLspServer(
        RocqLanguageService(REPO),
        StringIO("".join(_frame(message) for message in messages)),
        stdout,
    ).run()

    assert result == 0
    responses = _read_lsp_messages(stdout.getvalue())
    assert [response["id"] for response in responses] == list(range(1, 18))
    assert responses[0]["result"]["capabilities"]["hoverProvider"] is True
    assert responses[0]["result"]["capabilities"]["renameProvider"] == {
        "prepareProvider": True
    }
    assert responses[0]["result"]["capabilities"]["completionProvider"] == {
        "triggerCharacters": [".", "_"]
    }
    assert responses[0]["result"]["capabilities"]["codeLensProvider"] == {
        "resolveProvider": False
    }
    legend = responses[0]["result"]["capabilities"]["semanticTokensProvider"]["legend"]
    assert "function" in legend["tokenTypes"]
    assert "transition" in responses[1]["result"]["contents"]["value"]
    assert responses[2]["result"][0]["uri"] == uri
    assert responses[5]["result"][0]["name"] == "transition"
    assert responses[6]["result"]["signatures"][0]["label"].startswith("def transition")
    assert responses[7]["result"][0]["label"] == "transition"
    assert responses[8]["result"][0]["title"] == "Refresh Rocq extraction"
    assert responses[9]["result"][0]["data"]["symbol"] == "transition"
    assert len(responses[10]["result"]["data"]) % 5 == 0
    assert responses[11]["result"]["start"]["line"] == 66
    assert responses[12]["result"]["changes"]
    assert responses[14]["result"] is None
    assert responses[15]["error"]["code"] == -32603
    assert responses[16]["result"] is None
    assert RocqLspServer(RocqLanguageService(REPO), StringIO(""), StringIO()).run() == 0
    exit_out = StringIO()
    assert (
        RocqLspServer(
            RocqLanguageService(REPO),
            StringIO(_frame({"jsonrpc": "2.0", "id": 1, "method": "exit"})),
            exit_out,
        ).run()
        == 0
    )
    assert _read_lsp_messages(exit_out.getvalue())[0]["result"] is None
    assert (
        RocqLspServer(
            RocqLanguageService(REPO),
            StringIO("Header: value\r\n\r\n{}"),
            StringIO(),
        ).run()
        == 0
    )


def test_index_diagnostics_for_bad_maps_and_missing_declarations(
    tmp_path: Path,
) -> None:
    root = tmp_path
    models = root / "models"
    generated = root / "src" / "fido" / "rocq"
    models.mkdir()
    generated.mkdir(parents=True)
    (models / "toy.v").write_text("Definition toy := 1.\nDefinition dupe := 2.\n")
    (generated / "bad.pymap").write_text("{")
    pymap_header = (
        "stability,python_start_line,python_start_col,python_end_line,"
        "python_end_col,source_file,source_start_line,source_start_col,"
        "source_end_line,source_end_col,kind,symbol\n"
    )
    (generated / "wrong.pymap").write_text(
        pymap_header + "closed,1,0,1,0,toy.v,1,0,1,0,extraction,toy\n"
    )
    py_path = generated / "toy.py"
    py_path.write_text("def dupe() -> int:\n    return 2\n")
    os.utime(py_path, (1, 1))
    (generated / "toy.pymap").write_text(
        pymap_header
        + "open,1,0,1,3,toy.v,1,11,1,0,extraction,toy\n"
        + "open,1,0,1,0,toy.v,2,11,2,0,extraction,dupe\n"
        + "open,1,0,1,0,toy.v,2,11,2,0,extraction,dupe\n"
        + "open,1,0,1,0,toy.v,1,0,1,0,extraction,\n"
        + "open,1,0,1,0,missing.v,1,0,1,0,extraction,missing_source\n"
    )
    index = RocqIndex(root)

    index.refresh()

    messages = [diag.message for diag in index.diagnostics]
    assert any("bad source map" in message for message in messages)
    assert any("unsupported source map stability" in message for message in messages)
    assert any(
        "generated Python declaration missing for toy" in message
        for message in messages
    )
    assert any(
        "source file missing for missing_source" in message for message in messages
    )
    assert any("duplicate extracted symbol: dupe" in message for message in messages)
    assert any(
        "generated Python may be stale for dupe" in message for message in messages
    )
    assert index.diagnostics_for_file(models / "toy.v")
    assert any(
        node["path"] == "models/toy.v" for node in index.dependency_graph()["nodes"]
    )


def test_index_handles_missing_generated_dir_and_token_misses(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    index = RocqIndex(tmp_path)

    index.refresh()

    assert index.diagnostics[0].message == "generated Rocq Python directory is missing"
    assert index.symbol_at(tmp_path / "missing.v", 0, 0) is None
    source = tmp_path / "models" / "toy.v"
    source.write_text("Definition toy := 1.\n")
    assert index._token_at(source, -1, 0) is None
    assert index._token_at(source, 99, 0) is None
    assert index._token_at(source, 0, -1) is None
    assert index._token_at(source, 0, 99) is None
    assert index._token_at(source, 0, 0) == "Definition"
    assert index._token_at(source, 0, 10) is None
    assert index.semantic_tokens_for_file(source.with_name("missing.v")) == ()


def test_semantic_token_edge_cases() -> None:
    location = Location(
        REPO / "models" / "toy.v",
        Range(Position(1, 11), Position(1, 14)),
    )
    without_python = rocq_lsp.Symbol("Toy", location, None, None)
    with_python = rocq_lsp.Symbol(
        "toy",
        location,
        Location(REPO / "src" / "fido" / "rocq" / "toy.py", location.range),
        "def toy() -> int",
    )
    text = '(* outer (* nested *) comment *)\nDefinition toy := "unterminated \\'

    ranges = (
        *rocq_lsp._comment_and_string_ranges(text),
        *rocq_lsp._comment_and_string_ranges("(* unterminated"),
    )
    tokens = rocq_lsp._semantic_tokens_for_text(text, (without_python, with_python))
    encoded = rocq_lsp._encode_semantic_tokens(
        (
            rocq_lsp.SemanticToken(0, 1, 3, "keyword"),
            rocq_lsp.SemanticToken(0, 1, 3, "keyword"),
            rocq_lsp.SemanticToken(1, 0, 3, "enumMember", ("readonly",)),
        )
    )

    assert without_python.code_lens() is None
    assert with_python.code_lens() is not None
    assert any(item[3] == "comment" for item in ranges)
    assert any(item[3] == "string" for item in ranges)
    assert rocq_lsp._symbol_token_type("Toy") == "enumMember"
    assert rocq_lsp._symbol_token_type("toy") == "function"
    assert len(tokens) > 1
    assert len(encoded) == 15
    assert encoded[-1] == 2


def test_helpers_cover_comments_strings_uris_and_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    text = (
        'Definition keep := "escaped \\" keep". (* outer (* nested keep *) keep *)\n'
        "Definition keep := keep.\n"
    )
    matches = rocq_lsp._identifier_matches_without_comments(text, "keep")

    assert len(matches) == 3
    assert rocq_lsp._offset_to_position(text, matches[-1].start()) == (1, 19)
    assert rocq_lsp._path_from_uri("file:///tmp/a%20b.v") == Path("/tmp/a b.v")
    assert rocq_lsp._path_from_uri("models/session_lock.v") == Path(
        "models/session_lock.v"
    )
    assert not rocq_lsp._contains(Range(Position(0, 1), Position(0, 2)), 0, 0)
    assert rocq_lsp._contains(Range(Position(0, 1), Position(0, 2)), 0, 2)
    assert not rocq_lsp._contains(Range(Position(0, 1), Position(0, 2)), 0, 3)
    assert not rocq_lsp._contains(Range(Position(0, 1), Position(0, 2)), 1, 1)
    assert rocq_lsp._repo_path(REPO, Path("/tmp/outside.v")) == "/tmp/outside.v"
    assert rocq_lsp._python_signatures(tmp_path / "missing.py") == {}
    generated = tmp_path / "generated.py"
    generated.write_text(
        "answer: int = 42  # From model.v:1:1\n"
        "async def effect() -> int:  # From model.v:1:1\n"
        "    return 1\n"
        "plain = 'ok'  # From model.v:2:1\n"
        "target.attr = 0\n"
    )
    signatures = rocq_lsp._python_signatures(generated)
    assert signatures["answer"][0] == "answer: int = 42"
    assert signatures["effect"][0] == "async def effect() -> int"
    assert signatures["plain"][0] == "plain = 'ok'"
    assert "target" not in signatures
    assert rocq_lsp._signature_end_line(["def x("], 0) == 0
    assert rocq_lsp._strip_generated_comment("x  # From y.v:1:1") == "x"

    with pytest.raises(ValueError, match="unsupported command"):
        RocqLspCli(REPO, StringIO(), StringIO())._run_command(
            RocqLanguageService(REPO), _argparse_namespace("nope")
        )

    called: list[str] = []

    class FakeCli:
        def __init__(self, *_args: object) -> None:
            called.append("cli-init")

        def run(self, argv: list[str]) -> int:
            called.extend(argv)
            return 7

    class FakeServer:
        def __init__(self, *_args: object) -> None:
            called.append("server-init")

        def run(self) -> int:
            called.append("server-run")
            return 8

    monkeypatch.setattr(rocq_lsp, "RocqLspCli", FakeCli)
    monkeypatch.setattr(rocq_lsp, "RocqLspServer", FakeServer)
    monkeypatch.setattr(rocq_lsp.sys, "argv", ["prog", "diagnostics"])
    assert rocq_lsp.main_cli() == 7
    assert "diagnostics" in called
    assert rocq_lsp.main_lsp() == 8
    monkeypatch.setattr(rocq_lsp.sys, "argv", ["prog", "--stdio"])
    assert rocq_lsp.main() == 8
    monkeypatch.setattr(rocq_lsp.sys, "argv", ["prog", "diagnostics"])
    assert rocq_lsp.main() == 7


def test_index_symbol_at_falls_back_to_source_token_lookup(tmp_path: Path) -> None:
    source = tmp_path / "toy.v"
    source.write_text("foo\n")
    symbol = rocq_lsp.Symbol(
        name="foo",
        source=Location(
            source,
            Range(Position(1, 0), Position(1, 3)),
        ),
        python=None,
        python_signature=None,
    )
    index = RocqIndex(tmp_path)
    index._symbols = {"foo": symbol}

    assert index.symbol_at(source, 0, 1) == symbol


def _argparse_namespace(command: str) -> Any:
    return type("Args", (), {"command": command})()


def _frame(message: dict[str, Any]) -> str:
    body = json.dumps(message)
    return f"Content-Length: {len(body.encode())}\r\n\r\n{body}"


def _read_lsp_messages(raw: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    stream = StringIO(raw)
    while True:
        header = stream.readline()
        if not header:
            return messages
        length = int(header.partition(":")[2].strip())
        assert stream.readline() == "\r\n"
        messages.append(json.loads(stream.read(length)))
