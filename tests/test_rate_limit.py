"""Tests for fido.rate_limit — RateLimitMonitor + parsers (closes #812)."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from frozendict import frozendict

from fido.atomic import AtomicReference
from fido.rate_limit import (
    _REFRESH_INTERVAL,  # noqa: PLC2701
    RateLimitMonitor,
    RateLimitSnapshot,
    RateLimitWindow,
    _parse_window,  # noqa: PLC2701
)
from fido.registry import FidoState


def _make_state() -> AtomicReference[FidoState]:
    """Return a fresh :class:`~fido.atomic.AtomicReference` seeded with an
    empty :class:`~fido.registry.FidoState` for use in monitor tests."""
    return AtomicReference(FidoState(repos=frozendict()))


# ── _parse_window ─────────────────────────────────────────────────────────────


class TestParseWindow:
    def test_parses_full_payload(self) -> None:
        w = _parse_window(
            "core", {"used": 5, "limit": 5000, "reset": 1700000000, "remaining": 4995}
        )
        assert w.name == "core"
        assert w.used == 5
        assert w.limit == 5000
        assert w.resets_at == datetime.fromtimestamp(1700000000, tz=timezone.utc)

    def test_defaults_when_fields_missing(self) -> None:
        w = _parse_window("graphql", {})
        assert w.used == 0
        assert w.limit == 0
        assert w.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)

    def test_handles_garbage_reset_value(self) -> None:
        w = _parse_window("core", {"reset": "not-a-number"})
        assert w.resets_at == datetime.fromtimestamp(0, tz=timezone.utc)


# ── RateLimitWindow properties ────────────────────────────────────────────────


class TestRateLimitWindowProperties:
    def _w(self, used: int, limit: int) -> RateLimitWindow:
        return RateLimitWindow(
            name="core",
            used=used,
            limit=limit,
            resets_at=datetime(2026, 4, 19, tzinfo=timezone.utc),
        )

    def test_remaining_basic(self) -> None:
        assert self._w(used=10, limit=100).remaining == 90

    def test_remaining_clamps_to_zero_when_over(self) -> None:
        assert self._w(used=200, limit=100).remaining == 0

    def test_percent_remaining(self) -> None:
        assert self._w(used=25, limit=100).percent_remaining == 75.0

    def test_percent_remaining_zero_limit(self) -> None:
        assert self._w(used=0, limit=0).percent_remaining == 0.0


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
    def test_returns_none_before_first_refresh(self) -> None:
        gh = MagicMock()
        m = RateLimitMonitor(gh, _make_state())
        assert m.latest() is None
        gh.get_rate_limit.assert_not_called()

    def test_refresh_stores_snapshot(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        m = RateLimitMonitor(gh, _make_state())
        snap = m.refresh()
        assert snap is not None
        assert snap.rest.used == 5
        assert snap.graphql.used == 7
        assert m.latest() is snap

    def test_refresh_failure_keeps_prior_snapshot(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources(rest_used=10)
        m = RateLimitMonitor(gh, _make_state())
        first = m.refresh()
        gh.get_rate_limit.side_effect = RuntimeError("network down")
        second = m.refresh()
        assert second is None
        assert m.latest() is first

    def test_refresh_updates_when_new_data_arrives(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources(rest_used=10)
        m = RateLimitMonitor(gh, _make_state())
        m.refresh()
        gh.get_rate_limit.return_value = _resources(rest_used=42)
        m.refresh()
        latest = m.latest()
        assert latest is not None
        assert latest.rest.used == 42

    def test_handles_missing_resources_keys(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = {}
        m = RateLimitMonitor(gh, _make_state())
        snap = m.refresh()
        assert snap is not None
        assert snap.rest.limit == 0
        assert snap.graphql.limit == 0


class TestRateLimitMonitorStartThread:
    def test_returns_daemon_thread(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        t = RateLimitMonitor(gh, _make_state()).start_thread(_interval=60.0)
        assert t.daemon

    def test_thread_name(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        t = RateLimitMonitor(gh, _make_state()).start_thread(_interval=60.0)
        assert t.name == "rate-limit-monitor"

    def test_does_initial_refresh_before_loop(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        m = RateLimitMonitor(gh, _make_state())
        m.start_thread(_interval=60.0)
        # At least the inline initial refresh happened
        assert m.latest() is not None
        gh.get_rate_limit.assert_called()

    def test_calls_refresh_periodically(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        m = RateLimitMonitor(gh, _make_state())
        m.start_thread(_interval=0.01)
        time.sleep(0.1)
        assert gh.get_rate_limit.call_count >= 2


class TestRateLimitMonitorThreadSafety:
    """Smoke-test: refresh() and latest() don't deadlock or corrupt state
    when called concurrently from many threads (Python 3.14t free-threaded)."""

    def test_concurrent_refresh_and_latest(self) -> None:
        import threading

        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        m = RateLimitMonitor(gh, _make_state())
        m.refresh()  # seed

        stop = threading.Event()

        def writer() -> None:
            while not stop.is_set():
                m.refresh()

        def reader() -> None:
            while not stop.is_set():
                snap = m.latest()
                # always either None (briefly) or a real RateLimitSnapshot
                assert snap is None or isinstance(snap, RateLimitSnapshot)

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
