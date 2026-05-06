"""Tests for fido.rate_limit — RateLimitMonitor + parsers (closes #812)."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from frozendict import frozendict

from fido.atomic import AtomicReader, AtomicUpdater, create_atomic
from fido.provider import ProviderLimitWindow
from fido.rate_limit import (
    _REFRESH_INTERVAL,  # noqa: PLC2701
    GitHubLimit,
    RateLimitMonitor,
    _parse_window,  # noqa: PLC2701
)
from fido.registry import FidoState


def _make_state() -> tuple[AtomicReader[FidoState], AtomicUpdater[FidoState]]:
    """Return a fresh ``(reader, updater)`` pair seeded with an empty
    :class:`~fido.registry.FidoState` for use in monitor tests."""
    return create_atomic(FidoState(repos=frozendict(), github_limits=GitHubLimit()))


# ── _parse_window ─────────────────────────────────────────────────────────────


class TestParseWindow:
    def test_parses_full_payload(self) -> None:
        w = _parse_window(
            "rest", {"used": 5, "limit": 5000, "reset": 1700000000, "remaining": 4995}
        )
        assert w.name == "rest"
        assert w.used == 5
        assert w.limit == 5000
        assert w.resets_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    def test_defaults_when_fields_missing(self) -> None:
        w = _parse_window("graphql", {})
        assert w.used == 0
        assert w.limit == 0
        assert w.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)

    def test_handles_garbage_reset_value(self) -> None:
        w = _parse_window("rest", {"reset": "not-a-number"})
        assert w.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)


# ── ProviderLimitWindow properties ───────────────────────────────────────────


class TestProviderLimitWindowProperties:
    def _w(self, used: int, limit: int) -> ProviderLimitWindow:
        return ProviderLimitWindow(
            name="rest",
            used=used,
            limit=limit,
            resets_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        )

    def test_pressure_basic(self) -> None:
        assert self._w(used=10, limit=100).pressure == 0.1

    def test_pressure_none_when_limit_zero(self) -> None:
        assert self._w(used=0, limit=0).pressure is None


# ── GitHubLimit ───────────────────────────────────────────────────────────────


class TestGitHubLimit:
    def test_zero_value_has_none_used(self) -> None:
        gl = GitHubLimit()
        assert gl.rest.used is None
        assert gl.graphql.used is None

    def test_zero_value_window_names(self) -> None:
        gl = GitHubLimit()
        assert gl.rest.name == "rest"
        assert gl.graphql.name == "graphql"

    def test_custom_windows(self) -> None:
        rest = ProviderLimitWindow(name="rest", used=5, limit=5000)
        gql = ProviderLimitWindow(name="graphql", used=7, limit=5000)
        gl = GitHubLimit(rest=rest, graphql=gql)
        assert gl.rest.used == 5
        assert gl.graphql.used == 7


# ── RateLimitMonitor ──────────────────────────────────────────────────────────


def _resources(rest_used: int = 5, gql_used: int = 7) -> dict:
    return {
        "core": {
            "used": rest_used,
            "limit": 5000,
            "reset": 1700000000,
            "remaining": 5000 - rest_used,
        },
        "graphql": {
            "used": gql_used,
            "limit": 5000,
            "reset": 1700003600,
            "remaining": 5000 - gql_used,
        },
    }


class TestRateLimitMonitorRefresh:
    def test_state_has_zero_limits_before_first_refresh(self) -> None:
        gh = MagicMock()
        state_reader, state_updater = _make_state()
        RateLimitMonitor(gh, state_updater)
        # Monitor construction does not seed the state; zero-value until refresh
        assert state_reader.get().github_limits.rest.used is None
        gh.get_rate_limit.assert_not_called()

    def test_refresh_stores_snapshot_in_state(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        state_reader, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        limits = m.refresh()
        assert limits is not None
        assert limits.rest.used == 5
        assert limits.graphql.used == 7
        assert state_reader.get().github_limits is limits

    def test_refresh_failure_keeps_prior_state(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources(rest_used=10)
        state_reader, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        first = m.refresh()
        gh.get_rate_limit.side_effect = RuntimeError("network down")
        second = m.refresh()
        assert second is None
        assert state_reader.get().github_limits is first

    def test_refresh_updates_when_new_data_arrives(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources(rest_used=10)
        state_reader, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        m.refresh()
        gh.get_rate_limit.return_value = _resources(rest_used=42)
        m.refresh()
        assert state_reader.get().github_limits.rest.used == 42

    def test_handles_missing_resources_keys(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = {}
        _, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        limits = m.refresh()
        assert limits is not None
        assert limits.rest.limit == 0
        assert limits.graphql.limit == 0


class TestRateLimitMonitorStartThread:
    def test_returns_daemon_thread(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        _, state_updater = _make_state()
        t = RateLimitMonitor(gh, state_updater).start_thread(_interval=60.0)
        assert t.daemon

    def test_thread_name(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        _, state_updater = _make_state()
        t = RateLimitMonitor(gh, state_updater).start_thread(_interval=60.0)
        assert t.name == "rate-limit-monitor"

    def test_does_initial_refresh_before_loop(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        state_reader, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        m.start_thread(_interval=60.0)
        # At least the inline initial refresh happened
        limits = state_reader.get().github_limits
        assert limits.rest.used is not None
        assert limits.graphql.used is not None
        gh.get_rate_limit.assert_called()

    def test_calls_refresh_periodically(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        _, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        m.start_thread(_interval=0.01)
        time.sleep(0.1)
        assert gh.get_rate_limit.call_count >= 2


class TestRateLimitMonitorThreadSafety:
    """Smoke-test: refresh() and state reads don't deadlock or corrupt state
    when called concurrently from many threads (Python 3.14t free-threaded)."""

    def test_concurrent_refresh_and_latest(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        state_reader, state_updater = _make_state()
        m = RateLimitMonitor(gh, state_updater)
        m.refresh()  # seed

        stop = threading.Event()

        def writer() -> None:
            while not stop.is_set():
                m.refresh()

        def reader() -> None:
            while not stop.is_set():
                limits = state_reader.get().github_limits
                # always either the zero seed or a real populated snapshot
                assert isinstance(limits, GitHubLimit)

        threads = [threading.Thread(target=writer) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(4)
        ]
        for t in threads:
            t.start()
        time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join(timeout=2)
            assert not t.is_alive()


class TestRefreshInterval:
    def test_default_interval_is_one_minute(self) -> None:
        assert _REFRESH_INTERVAL == 60.0
