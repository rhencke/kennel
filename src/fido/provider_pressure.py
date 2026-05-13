"""Per-repo provider-pressure poller (closes #1696 parity).

Polls each unique provider configured across the managed repos every
60 seconds and publishes a fresh
:class:`~fido.appstate.ProviderPressureSnapshot` into
``RepoState.provider_pressure`` for each repo using that provider.

Mirrors :class:`~fido.rate_limit.RateLimitMonitor` in shape: write-only
relative to the snapshot (holds an
:class:`~fido.atomic.AtomicUpdater`, never an
:class:`~fido.atomic.AtomicReader`), single-writer poller thread, soft
fail on refresh errors (the prior snapshot stays put).

Dedup: providers are polled once per cycle keyed by
:class:`~fido.provider.ProviderID`; the result fans out to every repo
using that provider.  This matches the legacy
``provider_statuses_for_repo_configs`` semantics (one poll per provider,
not per repo).
"""

import logging
import threading
import time

from fido.appstate import (
    _EPOCH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderPressureSnapshot,
)
from fido.atomic import AtomicUpdater
from fido.config import RepoConfig
from fido.provider import ProviderID, ProviderPressureStatus
from fido.provider_factory import DefaultProviderFactory

log = logging.getLogger(__name__)

_REFRESH_INTERVAL: float = 60.0


def _snapshot_from_status(status: ProviderPressureStatus) -> ProviderPressureSnapshot:
    """Project a :class:`ProviderPressureStatus` (rich live object) into
    the JSON-friendly :class:`ProviderPressureSnapshot` carried on
    :class:`~fido.appstate.RepoState`.

    Pre-computes the ``level``/``warning``/``paused`` booleans so the
    wire format need not re-derive them.  ``None`` fields fall back to
    appstate sentinels (empty string, zero, epoch) — appstate types
    have no ``None``.
    """
    return ProviderPressureSnapshot(
        provider=str(status.provider),
        window_name=status.window_name or "",
        pressure=status.pressure if status.pressure is not None else 0.0,
        percent_used=status.percent_used if status.percent_used is not None else 0,
        resets_at=status.resets_at if status.resets_at is not None else _EPOCH,
        unavailable_reason=status.unavailable_reason or "",
        level=status.level,
        warning=status.warning,
        paused=status.paused,
    )


class ProviderPressureMonitor:
    """Polls each configured provider and publishes per-repo pressure snapshots.

    Construct with the configured repos, an
    :class:`~fido.atomic.AtomicUpdater` for :class:`FidoState`, and the
    same :class:`DefaultProviderFactory` used elsewhere (constructor DI
    per CLAUDE.md).  Single writer: only the poller thread calls
    :meth:`refresh`.  A failed per-provider poll logs the exception and
    leaves the prior snapshot in place; one provider's failure never
    blanks another provider's snapshot.
    """

    def __init__(
        self,
        repos: dict[str, RepoConfig],
        state: AtomicUpdater[FidoState],
        provider_factory: DefaultProviderFactory,
    ) -> None:
        self._repos = repos
        self._state = state
        self._factory = provider_factory

    def refresh(self) -> dict[ProviderID, ProviderPressureSnapshot]:
        """Hit each unique provider's limit endpoint and publish a fresh
        :class:`ProviderPressureSnapshot` to every repo using that
        provider.  Returns the per-provider snapshots dict (mostly for
        tests; the writer side has already published).
        """
        results: dict[ProviderID, ProviderPressureSnapshot] = {}
        # First pass: poll each unique provider once.  Any repo with
        # the same provider reuses the result.
        for repo_cfg in self._repos.values():
            if repo_cfg.provider in results:
                continue
            try:
                live = ProviderPressureStatus.from_snapshot(
                    self._factory.create_api(repo_cfg).get_limit_snapshot()
                )
            except Exception:
                log.exception(
                    "provider-pressure: %s refresh failed — keeping prior snapshot",
                    repo_cfg.provider,
                )
                continue
            results[repo_cfg.provider] = _snapshot_from_status(live)
        # Second pass: fan out to every repo using a successfully-polled
        # provider.  Repos whose provider failed this cycle keep their
        # prior snapshot.
        for name, repo_cfg in self._repos.items():
            snap = results.get(repo_cfg.provider)
            if snap is None:
                continue
            self._publish(name, snap)
        return results

    def _publish(self, name: str, snap: ProviderPressureSnapshot) -> None:
        """Pure lens write — pulled out so the closure binds *name* on
        the call frame, not the loop variable in :meth:`refresh` (avoids
        the B023 ``loop variable in lambda`` lint)."""
        self._state.update(lambda root: root.repos[name].provider_pressure, snap)

    def start_thread(self, *, _interval: float = _REFRESH_INTERVAL) -> threading.Thread:
        """Start a daemon thread that calls :meth:`refresh` every
        *_interval* seconds.

        Does an initial refresh inline before sleeping so ``fido
        status`` has data the moment the server is up.
        """
        self.refresh()

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.refresh()

        t = threading.Thread(
            target=_loop, daemon=True, name="provider-pressure-monitor"
        )
        t.start()
        return t


__all__ = ["ProviderPressureMonitor"]
