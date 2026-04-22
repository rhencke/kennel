import json
from io import StringIO
from pathlib import Path

import pytest

from fido import rocq_traceback
from fido.rocq_traceback import SourceMap, TracebackAnnotator, TracebackCLI


def _write_map(
    py_file: Path,
    *,
    start: int = 3,
    end: int = 5,
    version: int = 1,
) -> Path:
    map_file = py_file.with_suffix(".pymap")
    map_file.write_text(
        json.dumps(
            {
                "version": version,
                "python_file": py_file.name,
                "entries": [
                    {
                        "python_start_line": start,
                        "python_start_col": 0,
                        "python_end_line": end,
                        "python_end_col": 0,
                        "source_file": "source_maps.v",
                        "source_start_line": 12,
                        "source_start_col": 4,
                        "source_end_line": 13,
                        "source_end_col": 7,
                        "kind": "extraction",
                        "symbol": "source_map_runtime_error",
                    }
                ],
            }
        )
    )
    return map_file


class TestSourceMap:
    def test_loads_and_matches_smallest_range(self, tmp_path: Path) -> None:
        py_file = tmp_path / "boom.py"
        map_file = _write_map(py_file, start=1, end=10)
        data = json.loads(map_file.read_text())
        data["entries"].append(
            {
                "python_start_line": 4,
                "python_start_col": 0,
                "python_end_line": 4,
                "python_end_col": 0,
                "source_file": "inner.v",
                "source_start_line": 20,
                "source_start_col": 2,
            }
        )
        map_file.write_text(json.dumps(data))

        source_map = SourceMap.load(map_file)
        entry = source_map.lookup(4)

        assert entry is not None
        assert entry.rocq_location() == "inner.v:20:2"
        assert source_map.lookup(99) is None

    def test_rejects_unsupported_version(self, tmp_path: Path) -> None:
        map_file = _write_map(tmp_path / "boom.py", version=2)

        with pytest.raises(ValueError, match="unsupported source map version"):
            SourceMap.load(map_file)


class TestTracebackAnnotator:
    def test_annotates_matching_python_frame(self, tmp_path: Path) -> None:
        py_file = tmp_path / "source_map_runtime_error.py"
        _write_map(py_file)
        trace = (
            "Traceback (most recent call last):\n"
            f'  File "{py_file}", line 4, in source_map_runtime_error\n'
            'RuntimeError: "boom"'
        )
        err = StringIO()

        result = TracebackAnnotator(err).annotate(trace)

        assert f'File "{py_file}", line 4' in result
        assert "Rocq source: source_maps.v:12:4" in result
        assert err.getvalue() == ""

    def test_preserves_unmapped_frame(self, tmp_path: Path) -> None:
        trace = f'  File "{tmp_path / "missing.py"}", line 4, in f'

        result = TracebackAnnotator(StringIO()).annotate(trace)

        assert result == trace

    def test_warns_and_preserves_malformed_map(self, tmp_path: Path) -> None:
        py_file = tmp_path / "bad.py"
        py_file.with_suffix(".pymap").write_text("{")
        err = StringIO()
        trace = f'  File "{py_file}", line 1, in f'

        result = TracebackAnnotator(err).annotate(trace)

        assert result == trace
        assert "warning: could not read" in err.getvalue()

    def test_preserves_line_outside_map_range(self, tmp_path: Path) -> None:
        py_file = tmp_path / "boom.py"
        _write_map(py_file, start=10, end=20)
        trace = f'  File "{py_file}", line 3, in f'

        result = TracebackAnnotator(StringIO()).annotate(trace)

        assert result == trace


class TestTracebackCLI:
    def test_reads_stdin_and_preserves_trailing_newline(self, tmp_path: Path) -> None:
        py_file = tmp_path / "boom.py"
        _write_map(py_file)
        stdout = StringIO()

        exit_code = TracebackCLI(
            StringIO(f'  File "{py_file}", line 3, in f\n'),
            stdout,
            StringIO(),
        ).run([])

        assert exit_code == 0
        assert stdout.getvalue().endswith("\n")
        assert "source_maps.v:12:4" in stdout.getvalue()

    def test_reads_path_arguments(self, tmp_path: Path) -> None:
        py_file = tmp_path / "boom.py"
        trace_file = tmp_path / "trace.txt"
        _write_map(py_file)
        trace_file.write_text(f'  File "{py_file}", line 5, in f')
        stdout = StringIO()

        exit_code = TracebackCLI(StringIO(), stdout, StringIO()).run([str(trace_file)])

        assert exit_code == 0
        assert "Rocq source: source_maps.v:12:4" in stdout.getvalue()

    def test_main_uses_process_streams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdin = StringIO("plain text")
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(rocq_traceback.sys, "argv", ["traceback"])
        monkeypatch.setattr(rocq_traceback.sys, "stdin", stdin)
        monkeypatch.setattr(rocq_traceback.sys, "stdout", stdout)
        monkeypatch.setattr(rocq_traceback.sys, "stderr", stderr)

        assert rocq_traceback.main() == 0
        assert stdout.getvalue() == "plain text"
        assert stderr.getvalue() == ""
