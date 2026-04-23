import csv
from pathlib import Path

import pytest

from fido import rocq_pymap
from fido.rocq_pymap import PyMap, PyMapEntry, PyMapError

_HEADER = (
    "stability,python_start_line,python_start_col,python_end_line,python_end_col,"
    "source_file,source_start_line,source_start_col,source_end_line,"
    "source_end_col,kind,symbol,python_symbol\n"
)


def test_pymap_loads_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(_HEADER + "open,2,3,4,5,x.v,6,7,8,9,extraction,x,\n")

    source_map = PyMap.load(path)

    assert source_map.entries == (
        PyMapEntry(
            python_start_line=2,
            python_start_col=3,
            python_end_line=4,
            python_end_col=5,
            source_file="x.v",
            source_start_line=6,
            source_start_col=7,
            source_end_line=8,
            source_end_col=9,
            kind="extraction",
            symbol="x",
            python_symbol=None,
        ),
    )


def test_pymap_rejects_bad_integer(tmp_path: Path) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(_HEADER + "open,nope,3,4,5,x.v,6,7,8,9,extraction,x,\n")

    with pytest.raises(PyMapError, match="bad source map row"):
        PyMap.load(path)


def test_pymap_rejects_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(PyMapError, match="missing python_start_line"):
        PyMapEntry.from_row({"stability": "open"}, tmp_path / "x.pymap")


def test_pymap_rejects_unsupported_stability(tmp_path: Path) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(_HEADER + "closed,2,3,4,5,x.v,6,7,8,9,extraction,x,\n")

    with pytest.raises(PyMapError, match="unsupported source map stability"):
        PyMap.load(path)


def test_pymap_allows_extra_columns_for_open_stability(tmp_path: Path) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(
        _HEADER.rstrip("\n")
        + ",future\n"
        + "open,2,3,4,5,x.v,6,7,8,9,extraction,x,,ignored\n"
    )

    assert PyMap.load(path).entries[0].symbol == "x"


def test_pymap_wraps_csv_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "x.pymap"
    path.write_text(_HEADER)

    def fail_dict_reader(_: object) -> object:
        raise csv.Error("bad csv")

    monkeypatch.setattr(rocq_pymap.csv, "DictReader", fail_dict_reader)

    with pytest.raises(PyMapError, match="bad source map CSV"):
        PyMap.load(path)
