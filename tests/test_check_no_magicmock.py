"""Tests for tools/check_no_magicmock.py."""

import sys
from io import StringIO
from pathlib import Path

from tools.check_no_magicmock import check

# NOTE: this file is listed in _EXEMPTIONS so the checker doesn't flag the
# MagicMock strings written as fixture content below.

_VIOLATING_TESTS_CONTENT = "from unittest.mock import MagicMock\n"
_VIOLATING_SRC_CONTENT = "from unittest.mock import MagicMock\n"
_CLEAN_CONTENT = "# no dynamic mocks here\n"
_SRC_DOCSTRING_CONTENT = (
    '"""drive with hand-rolled fakes (no MagicMock per conventions)."""\nx = 1\n'
)


def _make_root(tmp_path: Path) -> Path:
    """Return a repo-root-shaped directory with tests/ and src/ sub-dirs."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "src").mkdir()
    return tmp_path


class TestCheckNoMagicmock:
    def test_returns_zero_when_no_tests_dir(self, tmp_path: Path) -> None:
        assert check(tmp_path, exemptions=frozenset()) == 0

    def test_returns_zero_when_no_src_dir(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_clean.py").write_text(_CLEAN_CONTENT)
        assert check(tmp_path, exemptions=frozenset()) == 0

    def test_returns_zero_for_clean_tree(self, tmp_path: Path) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_clean.py").write_text(_CLEAN_CONTENT)
        (root / "src" / "mod.py").write_text(_CLEAN_CONTENT)
        assert check(root, exemptions=frozenset()) == 0

    def test_flags_magicmock_in_test_file(self, tmp_path: Path, capsys: object) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_bad.py").write_text(_VIOLATING_TESTS_CONTENT)
        assert check(root, exemptions=frozenset()) == 1

    def test_violation_message_mentions_file_and_line(
        self, tmp_path: Path, capsys: object
    ) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_bad.py").write_text(_VIOLATING_TESTS_CONTENT)
        stderr_capture = StringIO()
        orig = sys.stderr
        sys.stderr = stderr_capture  # type: ignore[assignment]
        try:
            result = check(root, exemptions=frozenset())
        finally:
            sys.stderr = orig
        assert result == 1
        output = stderr_capture.getvalue()
        assert "tests/test_bad.py:1" in output
        assert "MagicMock" in output
        assert "constructor-DI" in output or "__init__" in output

    def test_exempted_file_is_skipped(self, tmp_path: Path) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_dirty.py").write_text(_VIOLATING_TESTS_CONTENT)
        assert check(root, exemptions=frozenset({"tests/test_dirty.py"})) == 0

    def test_flags_magicmock_import_in_src(self, tmp_path: Path) -> None:
        root = _make_root(tmp_path)
        (root / "src" / "bad.py").write_text(_VIOLATING_SRC_CONTENT)
        assert check(root, exemptions=frozenset()) == 1

    def test_does_not_flag_magicmock_in_src_docstring(self, tmp_path: Path) -> None:
        root = _make_root(tmp_path)
        (root / "src" / "ok.py").write_text(_SRC_DOCSTRING_CONTENT)
        assert check(root, exemptions=frozenset()) == 0

    def test_clean_tests_with_dirty_src_flags_one_violation(
        self, tmp_path: Path
    ) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_clean.py").write_text(_CLEAN_CONTENT)
        (root / "src" / "bad.py").write_text(_VIOLATING_SRC_CONTENT)
        assert check(root, exemptions=frozenset()) == 1

    def test_multiple_violations_all_reported(self, tmp_path: Path) -> None:
        root = _make_root(tmp_path)
        (root / "tests" / "test_a.py").write_text(_VIOLATING_TESTS_CONTENT)
        (root / "tests" / "test_b.py").write_text(_VIOLATING_TESTS_CONTENT)
        stderr_capture = StringIO()
        orig = sys.stderr
        sys.stderr = stderr_capture  # type: ignore[assignment]
        try:
            result = check(root, exemptions=frozenset())
        finally:
            sys.stderr = orig
        assert result == 1
        output = stderr_capture.getvalue()
        assert "test_a.py" in output
        assert "test_b.py" in output
        assert "2 violation(s)" in output
