"""GitHub rate-limit monitor for ``fido status`` (closes #812 follow-up).

Polls ``GET /rate_limit`` once per minute (per GitHub docs, this endpoint
itself does not count against any quota), and publishes the latest snapshot
into :class:`~fido.registry.FidoState` via a CAS update on the registry's
:class:`~fido.atomic.AtomicReference`.  Reads are lock-free: callers call
:meth:`~RateLimitMonitor.latest` which simply reads
:attr:`~fido.registry.FidoState.rate_limit` from the current snapshot.

The poller thread treats fetch failures as soft errors — the previous
snapshot stays put until the next successful refresh.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fido.atomic import AtomicReference
from fido.github import GitHub

log = logging.getLogger(__name__)

_REFRESH_INTERVAL: float = 60.0


@dataclass(frozen=True)
class RateLimitWindow:
    """One resource window from ``GET /rate_limit``.

    *resets_at* is the tz-aware datetime when *used* drops back to zero
    (parsed from GitHub's epoch-seconds ``reset`` field).
    """

    name: str
    used: int
    limit: int
    resets_at: datetime

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)

    @property
    def percent_remaining(self) -> float:
        if self.limit <= 0:
            return 0.0
        return 100.0 * self.remaining / self.limit


@dataclass(frozen=True)
class RateLimitSnapshot:
    """One full ``/rate_limit`` response, parsed for the windows we display."""

    rest: RateLimitWindow
    graphql: RateLimitWindow
    fetched_at: datetime


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

    def latest(self) -> RateLimitSnapshot | None:
        """Lock-free read of the latest rate-limit snapshot, or ``None``
        before the first successful refresh."""
        return self._state.get().rate_limit

    def refresh(self) -> RateLimitSnapshot | None:
        """Hit ``GET /rate_limit`` and publish the snapshot into
        :class:`~fido.registry.FidoState` via CAS update.

        Returns the new snapshot on success, ``None`` on failure (the
        prior snapshot remains in the shared state).
        """
        try:
            resources = self._gh.get_rate_limit()
        except Exception:
            log.exception("rate-limit monitor: refresh failed — keeping prior snapshot")
            return None
        snapshot = RateLimitSnapshot(
            rest=_parse_window("core", resources.get("core") or {}),
            graphql=_parse_window("graphql", resources.get("graphql") or {}),
            fetched_at=datetime.now(tz=timezone.utc),
        )
        self._state.update(lambda root: root.rate_limit, snapshot)
        log.info(
            "rate-limit: rest %d/%d, graphql %d/%d",
            snapshot.rest.used,
            snapshot.rest.limit,
            snapshot.graphql.used,
            snapshot.graphql.limit,
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


def _parse_window(name: str, raw: dict[str, Any]) -> RateLimitWindow:
    """Convert one ``resources.<name>`` entry into a :class:`RateLimitWindow`.

    Defaults to ``0/0`` and the unix epoch when fields are missing — the
    poller never raises just because GitHub omitted a field; the caller
    will see a zero-limit window and can ignore it.
    """
    reset_epoch = raw.get("reset", 0)
    try:
        resets_at = datetime.fromtimestamp(int(reset_epoch), tz=timezone.utc)
    except TypeError, ValueError, OverflowError, OSError:
        resets_at = datetime.fromtimestamp(0, tz=timezone.utc)
    return RateLimitWindow(
        name=name,
        used=int(raw.get("used", 0)),
        limit=int(raw.get("limit", 0)),
        resets_at=resets_at,
    )


__all__ = [
    "RateLimitMonitor",
    "RateLimitSnapshot",
    "RateLimitWindow",
]
