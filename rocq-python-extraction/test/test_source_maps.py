from __future__ import annotations

import io
import json
import traceback
from pathlib import Path

import pytest

from kennel.rocq_traceback import TracebackAnnotator


def test_generated_python_lines_carry_source_comments(build_default: Path) -> None:
    source = (build_default / "source_map_runtime_error.py").read_text()
    lines = [line for line in source.splitlines() if line]

    assert lines
    assert all("# From source_maps.v:" in line for line in lines)


def test_generated_pymap_points_back_to_rocq_source(build_default: Path) -> None:
    data = json.loads((build_default / "source_map_runtime_error.pymap").read_text())

    assert data["version"] == 1
    assert data["python_file"] == "source_map_runtime_error.py"
    assert data["entries"][0]["source_file"] == "source_maps.v"
    assert data["entries"][0]["source_start_line"] > 0
    assert data["entries"][0]["source_start_col"] >= 0


def test_traceback_cli_resolves_runtime_error_to_rocq_source(
    build_default: Path,
) -> None:
    import source_map_runtime_error

    with pytest.raises(RuntimeError) as exc_info:
        source_map_runtime_error.source_map_runtime_error(0)

    formatted = "".join(
        traceback.format_exception(
            exc_info.type,
            exc_info.value,
            exc_info.tb,
        )
    )

    annotated = TracebackAnnotator(err=io.StringIO()).annotate(formatted)

    assert "Rocq source: source_maps.v:" in annotated
