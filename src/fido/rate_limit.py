"""GitHub rate-limit monitor for ``fido status`` (closes #812 follow-up).

Polls ``GET /rate_limit`` once per minute (per GitHub docs, this endpoint
itself does not count against any quota), and publishes the latest
:class:`GitHubLimit` snapshot into
:attr:`~fido.registry.FidoState.github_limits` via a CAS update on the
registry's :class:`~fido.atomic.AtomicUpdater`.  Reads are lock-free:
callers read ``state.github_limits`` from the current
:class:`~fido.registry.FidoState`.

The poller thread treats fetch failures as soft errors — the previous
snapshot stays put until the next successful refresh.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fido.atomic import AtomicUpdater
from fido.github import GitHub
from fido.provider import ProviderLimitWindow

if TYPE_CHECKING:
    from fido.registry import FidoState

log = logging.getLogger(__name__)

_REFRESH_INTERVAL: float = 60.0

_ZERO_WINDOW_REST = ProviderLimitWindow(name="rest")
_ZERO_WINDOW_GRAPHQL = ProviderLimitWindow(name="graphql")


@dataclass(frozen=True)
class GitHubLimit:
    """Normalized GitHub platform rate-limit state (REST + GraphQL windows).

    The zero value (``GitHubLimit()``) is the initial sentinel — both
    windows have ``used=None``, meaning the monitor has not yet completed
    a successful poll.  After the first successful :meth:`RateLimitMonitor.refresh`
    the ``used`` fields will be integers (possibly ``0``).

    Stored at :attr:`~fido.registry.FidoState.github_limits`; updated
    atomically via :class:`~fido.atomic.AtomicUpdater`.
    """

    rest: ProviderLimitWindow = _ZERO_WINDOW_REST
    graphql: ProviderLimitWindow = _ZERO_WINDOW_GRAPHQL


class RateLimitMonitor:
    """Polls ``GET /rate_limit`` and publishes into the registry snapshot.

    Construct with a :class:`~fido.github.GitHub` client and an
    :class:`~fido.atomic.AtomicUpdater` for the registry's
    :class:`~fido.registry.FidoState` (constructor DI per CLAUDE.md).

    This object is write-only relative to the snapshot — it holds an
    :class:`~fido.atomic.AtomicUpdater`, not an
    :class:`~fido.atomic.AtomicReader`.  Status display reads
    ``registry.get_state().github_limits`` directly without going through
    the monitor.

    Either call :meth:`refresh` directly or hand the monitor to
    :meth:`start_thread` for the 60 s poller.  A failed refresh logs the
    exception and leaves the prior snapshot in place — ``fido status`` keeps
    showing the last known good numbers rather than blanking.

    Single writer: only the poller thread calls :meth:`refresh`.
    """

    def __init__(self, gh: GitHub, state: "AtomicUpdater[FidoState]") -> None:
        self._gh = gh
        self._state = state

    def refresh(self) -> "GitHubLimit | None":
        """Hit ``GET /rate_limit`` and publish the snapshot into
        :attr:`~fido.registry.FidoState.github_limits` via CAS update.

        Returns the new :class:`GitHubLimit` on success, ``None`` on failure
        (the prior snapshot remains in the shared state).
        """
        try:
            resources = self._gh.get_rate_limit()
        except Exception:
            log.exception("rate-limit monitor: refresh failed — keeping prior snapshot")
            return None
        limits = GitHubLimit(
            rest=_parse_window("rest", resources.get("core") or {}),
            graphql=_parse_window("graphql", resources.get("graphql") or {}),
        )
        self._state.update(lambda root: root.github_limits, limits)
        log.info(
            "rate-limit: rest %s/%s, graphql %s/%s",
            limits.rest.used,
            limits.rest.limit,
            limits.graphql.used,
            limits.graphql.limit,
        )
        return limits

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


__all__ = ["GitHubLimit", "RateLimitMonitor"]
