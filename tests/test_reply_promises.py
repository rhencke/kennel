import os
from pathlib import Path

import pytest

from fido.reply_promises import (
    ReplyPromise,
    add_reply_promise,
    list_reply_promises,
    remove_reply_promise,
)


def test_add_reply_promise_creates_empty_file(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    path = add_reply_promise(fido_dir, "pulls", 123)
    assert path == fido_dir / "reply-promises" / "pulls-123"
    assert path.exists()
    assert path.read_text() == ""


def test_add_reply_promise_is_idempotent(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    first = add_reply_promise(fido_dir, "issues", 55)
    first.write_text("")
    second = add_reply_promise(fido_dir, "issues", 55)
    assert first == second
    assert list_reply_promises(fido_dir) == [
        ReplyPromise(comment_type="issues", comment_id=55, path=first)
    ]


def test_remove_reply_promise_deletes_file(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    path = add_reply_promise(fido_dir, "pulls", 321)
    remove_reply_promise(fido_dir, "pulls", 321)
    assert not path.exists()


def test_remove_reply_promise_missing_file_is_noop(tmp_path: Path) -> None:
    # Regression for #665: the worker recovery path and the webhook reply
    # path can both try to remove the same promise.  Missing file means the
    # intent ("make sure it's gone") is already satisfied.
    remove_reply_promise(tmp_path / "fido", "pulls", 999)


def test_list_reply_promises_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    assert list_reply_promises(fido_dir) == []
    assert (fido_dir / "reply-promises").is_dir()


def test_list_reply_promises_sorts_by_mtime(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    first = add_reply_promise(fido_dir, "pulls", 1)
    second = add_reply_promise(fido_dir, "issues", 2)
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    assert list_reply_promises(fido_dir) == [
        ReplyPromise(comment_type="pulls", comment_id=1, path=first),
        ReplyPromise(comment_type="issues", comment_id=2, path=second),
    ]


def test_list_reply_promises_raises_on_directory_entry(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    promise_dir = fido_dir / "reply-promises"
    promise_dir.mkdir(parents=True)
    (promise_dir / "subdir").mkdir()
    with pytest.raises(IsADirectoryError):
        list_reply_promises(fido_dir)


def test_list_reply_promises_raises_on_bad_filename(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    promise_dir = fido_dir / "reply-promises"
    promise_dir.mkdir(parents=True)
    (promise_dir / "junk").touch()
    with pytest.raises(ValueError, match="invalid reply promise filename"):
        list_reply_promises(fido_dir)


def test_list_reply_promises_raises_on_bad_comment_type(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    promise_dir = fido_dir / "reply-promises"
    promise_dir.mkdir(parents=True)
    (promise_dir / "comments-7").touch()
    with pytest.raises(ValueError, match="invalid reply promise comment type"):
        list_reply_promises(fido_dir)


def test_list_reply_promises_raises_on_bad_comment_id(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    promise_dir = fido_dir / "reply-promises"
    promise_dir.mkdir(parents=True)
    (promise_dir / "pulls-nope").touch()
    with pytest.raises(ValueError, match="invalid reply promise filename"):
        list_reply_promises(fido_dir)


def test_add_reply_promise_raises_on_bad_comment_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid reply promise comment type"):
        add_reply_promise(tmp_path / "fido", "comments", 7)


def test_remove_reply_promise_raises_on_bad_comment_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid reply promise comment type"):
        remove_reply_promise(tmp_path / "fido", "comments", 7)
