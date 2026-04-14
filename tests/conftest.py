"""Shared pytest fixtures for kennel tests."""

from __future__ import annotations

import pytest

from kennel import claude


@pytest.fixture(autouse=True)
def _reset_claude_talker_registry():
    """Clear the global :class:`~kennel.claude.ClaudeTalker` registry between
    tests so entries from one test can't leak into the next and cause a
    spurious :class:`~kennel.claude.ClaudeLeakError`.  Also clears any
    thread-local repo_name the test may have set via
    :func:`kennel.claude.set_thread_repo`.
    """
    yield
    with claude._talkers_lock:
        claude._talkers.clear()
    claude.set_thread_repo(None)
