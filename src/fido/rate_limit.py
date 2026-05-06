"""GitHub rate-limit monitor for ``fido status`` (closes #812 follow-up).

Polls ``GET /rate_limit`` once per minute (per GitHub docs, this endpoint
itself does not count against any quota), and publishes the latest snapshot
into :attr:`~fido.registry.FidoState.provider_limits` under the
``"github"`` key via a CAS update on the registry's
:class:`~fido.atomic.AtomicReference`.  Reads are lock-free: callers read
``provider_limits["github"]`` from the current :class:`~fido.registry.FidoState`.

The poller thread treats fetch failures as soft errors — the previous
snapshot stays put until the next successful refresh.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fido.atomic import AtomicReference
from fido.github import GitHub
from fido.provider import ProviderID, ProviderLimitSnapshot, ProviderLimitWindow

log = logging.getLogger(__name__)

_REFRESH_INTERVAL: float = 60.0


class RateLimitMonitor:
    """Lock-free holder for the latest ``GET /rate_limit`` snapshot.

    Construct with a :class:`~fido.github.GitHub` client and the
    :class:`~fido.atomic.AtomicReference` backing
    :class:`~fido.registry.WorkerRegistry`'s
    :class:`~fido.registry.FidoState` (constructor DI per CLAUDE.md).
    Either call :meth:`refresh` directly or hand the monitor to
    :meth:`start_thread` for the 60s poller.

    A failed refresh logs the exception and leaves the prior snapshot in
    place — ``fido status`` keeps showing the last known good numbers
    rather than blanking.

    Single writer: only the poller thread calls :meth:`refresh`.  Reads
    are lock-free via :meth:`latest`.
    """

    def __init__(self, gh: GitHub, state: AtomicReference[Any]) -> None:
        self._gh = gh
        self._state = state
        # Seed the provider_limits map with a zero-value entry so the Lens
        # path exists before the first refresh completes.
        limits = self._state.get().provider_limits
        if ProviderID.GITHUB not in limits:
            zero = ProviderLimitSnapshot(provider=ProviderID.GITHUB)
            self._state.update(
                lambda root: root.provider_limits[ProviderID.GITHUB], zero
            )

    def latest(self) -> ProviderLimitSnapshot | None:
        """Lock-free read of the latest rate-limit snapshot, or ``None``
        before the first successful refresh."""
        return self._state.get().provider_limits.get(ProviderID.GITHUB)

    def refresh(self) -> ProviderLimitSnapshot | None:
        """Hit ``GET /rate_limit`` and publish the snapshot into
        :attr:`~fido.registry.FidoState.provider_limits` via CAS update.

        Returns the new snapshot on success, ``None`` on failure (the
        prior snapshot remains in the shared state).
        """
        try:
            resources = self._gh.get_rate_limit()
        except Exception:
            log.exception("rate-limit monitor: refresh failed — keeping prior snapshot")
            return None
        snapshot = ProviderLimitSnapshot(
            provider=ProviderID.GITHUB,
            windows=(
                _parse_window("rest", resources.get("core") or {}),
                _parse_window("graphql", resources.get("graphql") or {}),
            ),
        )
        self._state.update(
            lambda root: root.provider_limits[ProviderID.GITHUB], snapshot
        )
        rest = snapshot.windows[0] if snapshot.windows else None
        gql = snapshot.windows[1] if len(snapshot.windows) > 1 else None
        log.info(
            "rate-limit: rest %s/%s, graphql %s/%s",
            rest.used if rest else "?",
            rest.limit if rest else "?",
            gql.used if gql else "?",
            gql.limit if gql else "?",
        )
        return snapshot

    def start_thread(self, *, _interval: float = _REFRESH_INTERVAL) -> threading.Thread:
        """Start a daemon thread that calls :meth:`refresh` every
        *_interval* seconds.

        Does an initial refresh inline before sleeping so ``fido
        status`` can show numbers as soon as the server is up.
        """
        self.refresh()

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.refresh()

        t = threading.Thread(target=_loop, daemon=True, name="rate-limit-monitor")
        t.start()
        return t


def _parse_window(name: str, raw: dict[str, Any]) -> ProviderLimitWindow:
    """Convert one ``resources.<name>`` entry into a :class:`ProviderLimitWindow`.

    Defaults to ``0/0`` and the unix epoch when fields are missing — the
    poller never raises just because GitHub omitted a field; the caller
    will see a zero-limit window and can ignore it.
    """
    reset_epoch = raw.get("reset", 0)
    try:
        resets_at = datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
    except TypeError, ValueError, OverflowError, OSError:
        resets_at = datetime.fromtimestamp(0, tz=timezone.utc)
    return ProviderLimitWindow(
        name=name,
        used=int(raw.get("used", 0)),
        limit=int(raw.get("limit", 0)),
        resets_at=resets_at,
    )


__all__ = [
    "RateLimitMonitor",
]
