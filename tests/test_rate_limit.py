"""Tests for fido.rate_limit — RateLimitMonitor + parsers (closes #812)."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from frozendict import frozendict

from fido.atomic import AtomicReference
from fido.provider import ProviderID, ProviderLimitSnapshot, ProviderLimitWindow
from fido.rate_limit import (
    _REFRESH_INTERVAL,  # noqa: PLC2701
    RateLimitMonitor,
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
    def test_returns_zero_snapshot_before_first_refresh(self) -> None:
        gh = MagicMock()
        m = RateLimitMonitor(gh, _make_state())
        snap = m.latest()
        # Seeded with zero-value snapshot (no windows populated from API yet)
        assert snap is not None
        assert snap.provider == ProviderID.GITHUB
        assert snap.windows == ()
        gh.get_rate_limit.assert_not_called()

    def test_refresh_stores_snapshot(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = _resources()
        m = RateLimitMonitor(gh, _make_state())
        snap = m.refresh()
        assert snap is not None
        assert snap.provider == ProviderID.GITHUB
        rest_w = snap.windows[0]
        gql_w = snap.windows[1]
        assert rest_w.used == 5
        assert gql_w.used == 7
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
        assert latest.windows[0].used == 42

    def test_handles_missing_resources_keys(self) -> None:
        gh = MagicMock()
        gh.get_rate_limit.return_value = {}
        m = RateLimitMonitor(gh, _make_state())
        snap = m.refresh()
        assert snap is not None
        assert snap.windows[0].limit == 0
        assert snap.windows[1].limit == 0


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
        snap = m.latest()
        assert snap is not None
        assert len(snap.windows) == 2
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
                # always either the zero seed or a real populated snapshot
                assert snap is not None
                assert isinstance(snap, ProviderLimitSnapshot)

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
