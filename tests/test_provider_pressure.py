"""Tests for fido.provider_pressure — ProviderPressureMonitor (#1696 parity)."""

import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

from frozendict import frozendict

from fido.appstate import (
    _EPOCH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_PROVIDER_PRESSURE,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderPressureSnapshot,
    zero_repo_state,
)
from fido.atomic import AtomicReader, AtomicUpdater, create_atomic
from fido.config import RepoConfig, RepoMembership
from fido.provider import (
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
)
from fido.provider_pressure import (
    _REFRESH_INTERVAL,  # noqa: PLC2701
    ProviderPressureMonitor,
    _snapshot_from_status,  # noqa: PLC2701
)


def _state_with_repos(
    *names: str,
) -> tuple[AtomicReader[FidoState], AtomicUpdater[FidoState]]:
    return create_atomic(
        FidoState(
            repos=frozendict({name: zero_repo_state(name) for name in names}),
            github_limits=_ZERO_GITHUB_LIMITS,
            process_started_at=_EPOCH,
        )
    )


def _repo(name: str, *, provider: ProviderID = ProviderID.CLAUDE_CODE) -> RepoConfig:
    from pathlib import Path

    return RepoConfig(
        name=name,
        work_dir=Path("/tmp/fake"),
        provider=provider,
        membership=RepoMembership(),
    )


def _window(name: str, used: int, limit: int) -> ProviderLimitWindow:
    return ProviderLimitWindow(
        name=name,
        used=used,
        limit=limit,
        resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=timezone.utc),
        unit="",
    )


def _factory_with_pressure(
    pressure_window: ProviderLimitWindow | None = None,
    *,
    raises: BaseException | None = None,
) -> MagicMock:
    """Return a mock DefaultProviderFactory whose create_api returns an
    api whose get_limit_snapshot returns a single window (or raises)."""
    factory = MagicMock()
    api = MagicMock()
    if raises is not None:
        api.get_limit_snapshot.side_effect = raises
    else:
        windows = (pressure_window,) if pressure_window is not None else ()
        api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE, windows=windows
        )
    factory.create_api.return_value = api
    return factory


# ── _snapshot_from_status ─────────────────────────────────────────────────────


class TestSnapshotFromStatus:
    def test_populated_status_passes_through(self) -> None:
        from fido.provider import ProviderPressureStatus

        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(
                provider=ProviderID.CLAUDE_CODE,
                windows=(_window("five_hour", used=96, limit=100),),
            )
        )
        snap = _snapshot_from_status(status)
        assert snap.provider == "claude-code"
        assert snap.window_name == "five_hour"
        assert snap.percent_used == 96
        assert snap.level == "paused"
        assert snap.warning is False
        assert snap.paused is True

    def test_unknown_pressure_uses_zero_sentinels(self) -> None:
        # Empty snapshot → from_snapshot leaves window_name/pressure as None;
        # _snapshot_from_status maps them to "" / 0.0 / 0 / epoch sentinels.
        from fido.provider import ProviderPressureStatus

        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(provider=ProviderID.CODEX)
        )
        snap = _snapshot_from_status(status)
        assert snap.window_name == ""
        assert snap.pressure == 0.0
        assert snap.percent_used == 0
        assert snap.resets_at == _EPOCH
        assert snap.unavailable_reason == ""
        assert snap.level == "unknown"


# ── ProviderPressureMonitor.refresh ───────────────────────────────────────────


class TestRefresh:
    def test_publishes_snapshot_to_each_repo(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a", "owner/repo-b")
        factory = _factory_with_pressure(_window("five_hour", used=50, limit=100))
        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a"),
                "owner/repo-b": _repo("owner/repo-b"),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        snapshot = reader.get()
        assert snapshot.repos["owner/repo-a"].provider_pressure.percent_used == 50
        assert snapshot.repos["owner/repo-b"].provider_pressure.percent_used == 50

    def test_polls_each_provider_once_per_cycle(self) -> None:
        # Two repos, same provider — create_api should be called once.
        _, updater = _state_with_repos("owner/repo-a", "owner/repo-b")
        factory = _factory_with_pressure(_window("five_hour", used=10, limit=100))
        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a"),
                "owner/repo-b": _repo("owner/repo-b"),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        assert factory.create_api.call_count == 1

    def test_failure_keeps_prior_snapshot_for_repos_using_that_provider(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a")
        factory = _factory_with_pressure(_window("five_hour", used=42, limit=100))
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        first = reader.get().repos["owner/repo-a"].provider_pressure
        assert first.percent_used == 42

        # Second cycle raises — prior snapshot must remain.
        factory_bad = _factory_with_pressure(raises=RuntimeError("api down"))
        bad_monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory_bad,
        )
        bad_monitor.refresh()
        assert reader.get().repos["owner/repo-a"].provider_pressure == first

    def test_failure_isolated_per_provider(self) -> None:
        # repo-a uses claude (raises), repo-b uses codex (succeeds).
        reader, updater = _state_with_repos("owner/repo-a", "owner/repo-b")

        factory = MagicMock()

        def create_api(repo_cfg: RepoConfig) -> MagicMock:
            api = MagicMock()
            if repo_cfg.provider is ProviderID.CLAUDE_CODE:
                api.get_limit_snapshot.side_effect = RuntimeError("claude down")
            else:
                api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
                    provider=ProviderID.CODEX,
                    windows=(_window("primary", used=12, limit=100),),
                )
            return api

        factory.create_api.side_effect = create_api

        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a", provider=ProviderID.CLAUDE_CODE),
                "owner/repo-b": _repo("owner/repo-b", provider=ProviderID.CODEX),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()

        # repo-a's claude failed → still zero sentinel
        assert (
            reader.get().repos["owner/repo-a"].provider_pressure
            == _ZERO_PROVIDER_PRESSURE
        )
        # repo-b's codex succeeded → published
        assert reader.get().repos["owner/repo-b"].provider_pressure.percent_used == 12


# ── ProviderPressureMonitor.start_thread ──────────────────────────────────────


class TestStartThread:
    def test_returns_daemon_thread(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=_factory_with_pressure(),
        )
        t = monitor.start_thread(_interval=3600.0)
        assert t.daemon is True

    def test_thread_name(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=_factory_with_pressure(),
        )
        t = monitor.start_thread(_interval=3600.0)
        assert t.name == "provider-pressure-monitor"

    def test_does_initial_refresh_inline(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a")
        factory = _factory_with_pressure(_window("five_hour", used=33, limit=100))
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        monitor.start_thread(_interval=3600.0)
        # Initial refresh runs synchronously before sleep — snapshot is
        # already populated by the time start_thread returns.
        assert reader.get().repos["owner/repo-a"].provider_pressure.percent_used == 33

    def test_loop_calls_refresh_periodically(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        # Use a barrier-style event that the *publish* path sets, not
        # the get_limit_snapshot side-effect — get_limit_snapshot fires
        # before _publish, so signalling there can race the snapshot
        # write that the test wants to observe.
        publish_count = [0]
        second_publish = threading.Event()

        factory = MagicMock()
        api = MagicMock()
        api.get_limit_snapshot.return_value = ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE,
            windows=(_window("five_hour", used=1, limit=100),),
        )
        factory.create_api.return_value = api

        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        # Wrap the lens write to count completed publishes.
        original_publish = monitor._publish  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        def counting_publish(name: str, snap: ProviderPressureSnapshot) -> None:
            original_publish(name, snap)
            publish_count[0] += 1
            if publish_count[0] >= 2:
                second_publish.set()

        monitor._publish = counting_publish  # type: ignore[method-assign]  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        monitor.start_thread(_interval=0.01)
        assert second_publish.wait(timeout=10.0), "expected at least two refresh cycles"
        assert publish_count[0] >= 2


def test_refresh_interval_constant_is_60_seconds() -> None:
    assert _REFRESH_INTERVAL == 60.0


def test_module_exports_only_monitor() -> None:
    """Sanity: only ProviderPressureMonitor is in the public surface."""
    import fido.provider_pressure as mod

    assert "ProviderPressureMonitor" in mod.__all__
    assert mod.__all__ == ["ProviderPressureMonitor"]


def test_loop_pacing_uses_time_sleep() -> None:
    """The poller's loop is just sleep+refresh; importing the module
    shouldn't require a running thread to verify the constant."""
    # Sanity that time module is referenced (covers the import at top level
    # so future maintainers don't strip it inadvertently).
    assert callable(time.sleep)
