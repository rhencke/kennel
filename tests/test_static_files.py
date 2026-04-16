from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

from kennel.static_files import StaticFiles


class TestServeBasic:
    def test_serves_css_file(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("body { color: red; }")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", None, None)
        assert resp is not None
        assert resp.status == 200
        assert resp.body == b"body { color: red; }"
        headers = dict(resp.headers)
        assert headers["Content-Type"] == "text/css; charset=utf-8"
        assert headers["Content-Length"] == str(len(resp.body))

    def test_serves_xsl_file(self, tmp_path: Path) -> None:
        (tmp_path / "status.xsl").write_text("<xsl:stylesheet/>")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/status.xsl", None, None)
        assert resp is not None
        assert resp.status == 200
        assert resp.body == b"<xsl:stylesheet/>"
        headers = dict(resp.headers)
        assert headers["Content-Type"] == "application/xslt+xml; charset=utf-8"

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        sf = StaticFiles(tmp_path)
        assert sf.serve("/static/nope.css", None, None) is None

    def test_returns_none_for_unknown_extension(self, tmp_path: Path) -> None:
        (tmp_path / "secrets.txt").write_text("oh no")
        sf = StaticFiles(tmp_path)
        assert sf.serve("/static/secrets.txt", None, None) is None

    def test_returns_none_for_empty_path(self, tmp_path: Path) -> None:
        sf = StaticFiles(tmp_path)
        assert sf.serve("/static/", None, None) is None

    def test_returns_none_for_path_traversal(self, tmp_path: Path) -> None:
        # Create a file outside the static root
        (tmp_path / "outside.css").write_text("naughty")
        sub = tmp_path / "static"
        sub.mkdir()
        sf = StaticFiles(sub)
        assert sf.serve("/static/../outside.css", None, None) is None

    def test_returns_none_for_directory(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        sf = StaticFiles(tmp_path)
        assert sf.serve("/static/sub", None, None) is None


class TestCachingHeaders:
    def test_etag_present(self, tmp_path: Path) -> None:
        content = b"body { color: blue; }"
        (tmp_path / "style.css").write_bytes(content)
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", None, None)
        assert resp is not None
        headers = dict(resp.headers)
        expected_etag = '"' + hashlib.sha256(content).hexdigest()[:16] + '"'
        assert headers["ETag"] == expected_etag

    def test_last_modified_present(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", None, None)
        assert resp is not None
        headers = dict(resp.headers)
        assert "Last-Modified" in headers

    def test_cache_control_present(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", None, None)
        assert resp is not None
        headers = dict(resp.headers)
        assert headers["Cache-Control"] == "public, max-age=3600"


class TestConditionalRequests:
    def test_etag_match_returns_304(self, tmp_path: Path) -> None:
        content = b"body { color: red; }"
        (tmp_path / "style.css").write_bytes(content)
        sf = StaticFiles(tmp_path)
        etag = '"' + hashlib.sha256(content).hexdigest()[:16] + '"'
        resp = sf.serve("/static/style.css", etag, None)
        assert resp is not None
        assert resp.status == 304
        assert resp.body == b""
        headers = dict(resp.headers)
        assert headers["ETag"] == etag
        assert headers["Cache-Control"] == "public, max-age=3600"

    def test_etag_mismatch_returns_200(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", '"stale"', None)
        assert resp is not None
        assert resp.status == 200

    def test_if_modified_since_not_modified(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        # Use a date far in the future so the file is definitely not newer
        future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        since = format_datetime(future, usegmt=True)
        resp = sf.serve("/static/style.css", None, since)
        assert resp is not None
        assert resp.status == 304
        assert resp.body == b""

    def test_if_modified_since_modified(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        # Use a date far in the past so the file is newer
        past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        since = format_datetime(past, usegmt=True)
        resp = sf.serve("/static/style.css", None, since)
        assert resp is not None
        assert resp.status == 200

    def test_if_modified_since_malformed_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/style.css", None, "not-a-date")
        assert resp is not None
        assert resp.status == 200

    def test_etag_takes_precedence_over_if_modified_since(self, tmp_path: Path) -> None:
        content = b"body { color: red; }"
        (tmp_path / "style.css").write_bytes(content)
        sf = StaticFiles(tmp_path)
        etag = '"' + hashlib.sha256(content).hexdigest()[:16] + '"'
        # Etag matches → 304, even though If-Modified-Since is in the past
        past = datetime(2000, 1, 1, tzinfo=timezone.utc)
        since = format_datetime(past, usegmt=True)
        resp = sf.serve("/static/style.css", etag, since)
        assert resp is not None
        assert resp.status == 304


class TestUrlDecoding:
    def test_percent_encoded_filename(self, tmp_path: Path) -> None:
        (tmp_path / "my style.css").write_text("x")
        sf = StaticFiles(tmp_path)
        resp = sf.serve("/static/my%20style.css", None, None)
        assert resp is not None
        assert resp.status == 200
