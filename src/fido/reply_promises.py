"""Durable per-comment reply promises stored as empty files."""

from dataclasses import dataclass
from pathlib import Path

_COMMENT_TYPES = frozenset({"issues", "pulls"})


@dataclass(frozen=True)
class ReplyPromise:
    """One owed reply keyed only by comment type and comment id."""

    comment_type: str
    comment_id: int
    path: Path


def _promise_dir(fido_dir: Path) -> Path:
    return fido_dir / "reply-promises"


def _promise_path(fido_dir: Path, comment_type: str, comment_id: int) -> Path:
    return _promise_dir(fido_dir) / f"{comment_type}-{comment_id}"


def _validate_comment_type(comment_type: str) -> str:
    if comment_type not in _COMMENT_TYPES:
        raise ValueError(f"invalid reply promise comment type: {comment_type!r}")
    return comment_type


def _parse_promise_name(name: str) -> tuple[str, int]:
    comment_type, sep, raw_id = name.partition("-")
    if sep != "-":
        raise ValueError(f"invalid reply promise filename: {name!r}")
    _validate_comment_type(comment_type)
    try:
        comment_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"invalid reply promise filename: {name!r}") from exc
    return comment_type, comment_id


def add_reply_promise(fido_dir: Path, comment_type: str, comment_id: int) -> Path:
    """Create the durable promise file if it does not already exist."""
    _validate_comment_type(comment_type)
    path = _promise_path(fido_dir, comment_type, comment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def remove_reply_promise(fido_dir: Path, comment_type: str, comment_id: int) -> None:
    """Delete the durable promise file if present.

    The worker's recovery path and the webhook reply path can both race to
    remove the same promise (see #665).  The caller's intent is "make sure
    this promise is gone", so a missing file is success, not an error.
    """
    _validate_comment_type(comment_type)
    _promise_path(fido_dir, comment_type, comment_id).unlink(missing_ok=True)


def list_reply_promises(fido_dir: Path) -> list[ReplyPromise]:
    """Return promises in filesystem timestamp order."""
    promise_dir = _promise_dir(fido_dir)
    promise_dir.mkdir(parents=True, exist_ok=True)
    result: list[ReplyPromise] = []
    for path in promise_dir.iterdir():
        if not path.is_file():
            raise IsADirectoryError(f"reply promise entry is not a file: {path}")
        comment_type, comment_id = _parse_promise_name(path.name)
        result.append(
            ReplyPromise(
                comment_type=comment_type,
                comment_id=comment_id,
                path=path,
            )
        )
    result.sort(key=lambda item: item.path.stat().st_mtime_ns)
    return result
