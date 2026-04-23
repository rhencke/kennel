import io
import traceback
from pathlib import Path

import pytest

from fido.rocq_pymap import PyMap
from fido.rocq_traceback import TracebackAnnotator


def test_generated_python_has_no_inline_source_comments(build_default: Path) -> None:
    source = (build_default / "source_map_runtime_error.py").read_text()
    assert "# From source_maps.v:" not in source


def test_generated_pymap_points_back_to_rocq_source(build_default: Path) -> None:
    source_map = PyMap.load(build_default / "source_map_runtime_error.pymap")
    entry = source_map.entries[0]

    assert entry.source_file == "source_maps.v"
    assert entry.source_start_line > 0
    assert entry.source_start_col >= 0
    assert entry.python_start_line > 0
    assert entry.python_start_col >= 0
    assert entry.python_end_line >= entry.python_start_line
    assert entry.python_end_col >= entry.python_start_col


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
