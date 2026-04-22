"""Annotate extracted Python tracebacks with Rocq source locations."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

_FRAME_RE = re.compile(
    r'^(?P<prefix>\s*File ")(?P<file>[^"]+)(?P<mid>", line )'
    r"(?P<line>\d+)(?P<suffix>.*)$"
)


@dataclass(frozen=True)
class SourceMapEntry:
    python_start_line: int
    python_start_col: int
    python_end_line: int
    python_end_col: int
    source_file: str
    source_start_line: int
    source_start_col: int

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SourceMapEntry:
        return cls(
            python_start_line=int(data["python_start_line"]),
            python_start_col=int(data.get("python_start_col", 0)),
            python_end_line=int(data["python_end_line"]),
            python_end_col=int(data.get("python_end_col", 0)),
            source_file=str(data["source_file"]),
            source_start_line=int(data["source_start_line"]),
            source_start_col=int(data["source_start_col"]),
        )

    def contains(self, line: int) -> bool:
        return self.python_start_line <= line <= self.python_end_line

    def rocq_location(self) -> str:
        return f"{self.source_file}:{self.source_start_line}:{self.source_start_col}"


class SourceMap:
    def __init__(self, entries: tuple[SourceMapEntry, ...]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, path: Path) -> SourceMap:
        raw = json.loads(path.read_text())
        if raw.get("version") != 1:
            raise ValueError(f"unsupported source map version in {path}")
        entries = tuple(
            SourceMapEntry.from_json(entry) for entry in raw.get("entries", [])
        )
        return cls(entries)

    def lookup(self, line: int) -> SourceMapEntry | None:
        matches = [entry for entry in self._entries if entry.contains(line)]
        if not matches:
            return None
        return min(
            matches,
            key=lambda entry: entry.python_end_line - entry.python_start_line,
        )


class TracebackAnnotator:
    def __init__(self, err: IO[str]) -> None:
        self._err = err
        self._maps: dict[Path, SourceMap | None] = {}

    def annotate(self, text: str) -> str:
        return "\n".join(self._annotate_line(line) for line in text.splitlines())

    def _annotate_line(self, line: str) -> str:
        match = _FRAME_RE.match(line)
        if match is None:
            return line

        path = Path(match.group("file"))
        source_map = self._source_map_for(path)
        if source_map is None:
            return line

        entry = source_map.lookup(int(match.group("line")))
        if entry is None:
            return line

        return f"{line}\n    Rocq source: {entry.rocq_location()}"

    def _source_map_for(self, python_path: Path) -> SourceMap | None:
        map_path = python_path.with_suffix(".pymap")
        if map_path not in self._maps:
            self._maps[map_path] = self._load_source_map(map_path)
        return self._maps[map_path]

    def _load_source_map(self, map_path: Path) -> SourceMap | None:
        if not map_path.is_file():
            return None
        try:
            return SourceMap.load(map_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {map_path}: {exc}", file=self._err)
            return None


class TracebackCLI:
    def __init__(self, stdin: IO[str], stdout: IO[str], stderr: IO[str]) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr

    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(
            prog="traceback",
            description="Annotate extracted Python tracebacks with Rocq locations.",
        )
        parser.add_argument("paths", nargs="*")
        args = parser.parse_args(argv)

        annotator = TracebackAnnotator(self._stderr)
        for text in self._input_texts(args.paths):
            self._stdout.write(annotator.annotate(text))
            if text.endswith("\n"):
                self._stdout.write("\n")
        return 0

    def _input_texts(self, paths: list[str]) -> tuple[str, ...]:
        if not paths:
            return (self._stdin.read(),)
        return tuple(Path(path).read_text() for path in paths)


def main() -> int:
    return TracebackCLI(sys.stdin, sys.stdout, sys.stderr).run(sys.argv[1:])
