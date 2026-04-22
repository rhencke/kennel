"""Shared pytest fixtures for fido tests."""

import pytest

from fido import provider


@pytest.fixture(autouse=True)
def _reset_claude_talker_registry():
    """Clear the global :class:`~fido.provider.SessionTalker` registry between
    tests so entries from one test can't leak into the next and cause a
    spurious :class:`~fido.provider.SessionLeakError`.  Also clears any
    thread-local repo_name the test may have set via
    :func:`fido.provider.set_thread_repo`.
    """
    yield
    with provider._talkers_lock:
        provider._talkers.clear()
    provider.set_thread_repo(None)
