"""Static file serving with etag, last-modified, and cache-control caching."""

import dataclasses
import hashlib
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote

_CONTENT_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".xsl": "application/xslt+xml; charset=utf-8",
}

_MAX_AGE = 3600


@dataclasses.dataclass
class StaticFileResponse:
    """Response from static file serving — status, headers, and body."""

    status: int
    headers: list[tuple[str, str]]
    body: bytes


class StaticFiles:
    """Serves files from a directory root with HTTP caching headers.

    Supports ``ETag`` (content-hash), ``Last-Modified`` (filesystem mtime),
    and ``Cache-Control`` headers.  Honors ``If-None-Match`` and
    ``If-Modified-Since`` conditional requests with 304 Not Modified.

    Injected into ``WebhookHandler`` at the composition root.  Tests construct
    with a ``tmp_path`` containing real files — no filesystem mocking needed.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def serve(
        self,
        url_path: str,
        if_none_match: str | None,
        if_modified_since: str | None,
    ) -> StaticFileResponse | None:
        """Serve a static file, or return ``None`` if not found / not allowed.

        Only serves files with recognized extensions directly under the root.
        Returns 304 responses when conditional headers match.
        """
        relative = unquote(url_path.removeprefix("/static/"))
        if not relative:
            return None

        full_path = (self._root / relative).resolve()
        if not full_path.is_relative_to(self._root):
            return None

        if not full_path.is_file():
            return None

        content_type = _CONTENT_TYPES.get(full_path.suffix)
        if content_type is None:
            return None

        content = full_path.read_bytes()
        etag = '"' + hashlib.sha256(content).hexdigest()[:16] + '"'
        # Truncate to second precision — HTTP dates don't carry sub-second.
        mtime = datetime.fromtimestamp(int(full_path.stat().st_mtime), tz=timezone.utc)
        last_modified = format_datetime(mtime, usegmt=True)

        cache_control = f"public, max-age={_MAX_AGE}"

        if if_none_match == etag:
            return StaticFileResponse(
                status=304,
                headers=[("ETag", etag), ("Cache-Control", cache_control)],
                body=b"",
            )

        if if_modified_since:
            try:
                since = parsedate_to_datetime(if_modified_since)
                if mtime <= since:
                    return StaticFileResponse(
                        status=304,
                        headers=[
                            ("ETag", etag),
                            ("Last-Modified", last_modified),
                            ("Cache-Control", cache_control),
                        ],
                        body=b"",
                    )
            except ValueError, TypeError:
                pass

        return StaticFileResponse(
            status=200,
            headers=[
                ("Content-Type", content_type),
                ("Content-Length", str(len(content))),
                ("ETag", etag),
                ("Last-Modified", last_modified),
                ("Cache-Control", cache_control),
            ],
            body=content,
        )
