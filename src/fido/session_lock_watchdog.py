"""Session-lock watchdog — evict wedged FSM lock holders.

The session-lock FSM (extracted from ``models/session_lock.v``) proves
**safety** through ``no_dual_ownership`` and ``release_only_by_owner``,
but in earlier model revisions the only path out of an owned state was
the holder firing its voluntary ``WorkerRelease`` / ``HandlerRelease``
event.  When a holder thread parked inside
:meth:`~fido.claude.ClaudeSession.consume_until_result` on a subprocess
that streamed events forever (so ``idle_timeout`` never tripped) and
never produced ``type=result``, the holder never returned from
``with self:``, ``__exit__`` never ran, ``_fsm_release`` never fired,
and the FSM lock leaked indefinitely (closes #1377).

The model now also proves **liveness** via ``force_release_to_free``
and ``every_state_reaches_free``: ``ForceRelease`` is accepted in every
state and always lands in :class:`~fido.rocq.transition.Free`.  This
watchdog is the runtime driver that fires that event when a holder
has held the lock past a deadline — guarding the property that no
acquire can wait forever for a holder that will never release.

Per repo, every :data:`_WATCHDOG_INTERVAL` seconds:

1. Read the current :class:`~fido.provider.SessionTalker` snapshot.
2. If a holder is registered and ``now - talker.started_at >
   hold_deadline_seconds``, log a warning and call
   :meth:`~fido.provider.OwnedSession.force_release` with a reason
   string identifying the evicted tid and the elapsed hold time.
3. The provider's :meth:`_on_force_release` subclass hook (e.g.
   :meth:`~fido.claude.ClaudeSession._on_force_release`) knocks the
   wedged thread out of its parked IO call by killing the subprocess.

The watchdog itself never touches the FSM lock or the subprocess
directly — every action goes through the modeled
:meth:`force_release` API so the Rocq oracle catches any divergence.
"""

import logging
import threading
import time

from fido.config import RepoConfig
from fido.provider import OwnedSession, get_talker, talker_now
from fido.registry import WorkerRegistry

log = logging.getLogger(__name__)

# Seconds between watchdog ticks.  Eviction granularity is therefore
# ``_WATCHDOG_INTERVAL`` past the holder's deadline, in the worst case.
_WATCHDOG_INTERVAL: float = 30.0

# Default per-turn hold deadline.  A worker turn over this threshold is
# already a code smell (typical legitimate turns finish in seconds to
# a few minutes); past 15 minutes we assume the holder is wedged.
# Configurable per session for tests and tightening over time.
_DEFAULT_HOLD_DEADLINE_SECONDS: float = 900.0


class SessionLockWatchdog:
    """Per-repo poller that evicts FSM lock holders past a deadline.

    Accepts *registry* and *repos* via the constructor so tests can
    drive it directly with hand-rolled fakes (no MagicMock per
    fido test conventions).  *hold_deadline_seconds* is configurable
    per instance: tests use a tiny value so a single iteration triggers
    eviction; production uses :data:`_DEFAULT_HOLD_DEADLINE_SECONDS`.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
        *,
        hold_deadline_seconds: float = _DEFAULT_HOLD_DEADLINE_SECONDS,
    ) -> None:
        self.registry = registry
        self.repos = repos
        self.hold_deadline_seconds = hold_deadline_seconds

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0.

        For each configured repo with an attached session and an
        active holder, compute the hold age via
        :func:`~fido.provider.talker_now` and the talker's
        ``started_at`` timestamp.  If the age exceeds
        :attr:`hold_deadline_seconds`, fire
        :meth:`~fido.provider.OwnedSession.force_release` on the
        session.

        Repos without a session (provider not yet initialised, or a
        registry stub during boot) and repos with no current holder
        (FSM is :class:`~fido.rocq.transition.Free`) are silently
        skipped — both are normal idle states.
        """
        for repo_name in self.repos:
            session = self._resolve_session(repo_name)
            if session is None:
                continue
            talker = get_talker(repo_name)
            if talker is None:
                continue
            held_for = (talker_now() - talker.started_at).total_seconds()
            if held_for <= self.hold_deadline_seconds:
                continue
            reason = (
                f"held > {self.hold_deadline_seconds:.0f}s "
                f"by tid={talker.thread_id} kind={talker.kind} "
                f"(actual {held_for:.0f}s, description={talker.description!r})"
            )
            log.warning(
                "session-lock-watchdog[%s]: holder past deadline — %s",
                repo_name,
                reason,
            )
            session.force_release(reason=reason)
        return 0

    def _resolve_session(self, repo_name: str) -> OwnedSession | None:
        """Return the :class:`OwnedSession` for *repo_name*, or ``None``.

        :class:`~fido.registry.WorkerRegistry`'s
        :meth:`~fido.registry.WorkerRegistry.get_session` returns the
        provider session attached to the repo, or ``None`` when the
        worker has not constructed one yet.  We also accept a
        non-:class:`OwnedSession` return as ``None`` (defensive against
        future provider types that may not extend the base) — the
        watchdog then has nothing to do for that repo this tick.
        """
        session = self.registry.get_session(repo_name)
        if isinstance(session, OwnedSession):
            return session
        return None

    def start_thread(
        self, *, _interval: float = _WATCHDOG_INTERVAL
    ) -> threading.Thread:
        """Start a daemon thread that runs :meth:`run` every *_interval* seconds."""

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.run()

        t = threading.Thread(target=_loop, daemon=True, name="session-lock-watchdog")
        t.start()
        return t


def run(
    registry: WorkerRegistry,
    repos: dict[str, RepoConfig],
    *,
    hold_deadline_seconds: float = _DEFAULT_HOLD_DEADLINE_SECONDS,
) -> int:
    """Module-level entry point — create a watchdog and run one tick."""
    return SessionLockWatchdog(
        registry, repos, hold_deadline_seconds=hold_deadline_seconds
    ).run()
