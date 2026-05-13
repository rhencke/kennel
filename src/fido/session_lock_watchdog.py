"""Session-lock watchdog — evict wedged FSM lock holders.

The session-lock FSM (extracted from ``models/session_lock.v``) proves
**safety** through ``no_dual_ownership`` and ``release_only_by_owner``,
but in earlier model revisions the only path out of an owned state was
the holder firing its voluntary ``WorkerRelease`` / ``HandlerRelease``
event.  When a holder thread parked inside
:meth:`~fido.claude.ClaudeSession.consume_until_result` on a subprocess
that wedged before producing ``type=result``, the holder never returned
from ``with self:``, ``__exit__`` never ran, ``_fsm_release`` never
fired, and the FSM lock leaked indefinitely (closes #1377).

The model now also proves **liveness** via ``force_release_to_free``
and ``every_state_reaches_free``: ``ForceRelease`` is accepted in every
state and always lands in :class:`~fido.rocq.transition.Free`.  This
watchdog is the runtime driver that fires that event when a holder
has held the lock without receiving any data from the provider
subprocess for too long — guarding the property that no acquire can
wait forever for a holder that will never release.

Trigger semantics — **SYN without ACK** (closes #1709):

* Send arms the clock — setting
  :attr:`~fido.provider.OwnedSession.outstanding_send_at` to now.
* Receive disarms it — clearing the field back to ``None``.
* Multiple sends without an intervening receive simply restart the
  clock; the most recent unanswered send wins.
* Idle sessions never trigger (clock is ``None``).
* Forever-streaming sessions never trigger (each receive disarms).
* Only the **prompt-sent-no-reply** wedge fires: the clock has been
  armed continuously for more than ``no_reply_seconds``.

Per repo, every :data:`_WATCHDOG_INTERVAL` seconds:

1. Read :attr:`~fido.provider.OwnedSession.outstanding_send_at`.
2. If it is not ``None`` and ``now - outstanding_send_at >
   no_reply_seconds``, log a warning and call
   :meth:`~fido.provider.OwnedSession.force_release` with a reason
   string identifying the evicted tid and the silence duration.
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

# Default no-reply deadline: kill the holder when an outstanding send
# has been waiting for a receive for more than this long.  One hour is
# generous — the in-process ``idle_timeout`` (30 min) is the first line
# of defense for the totally-silent subprocess; this watchdog is the
# safety net for the SYN-without-ACK case where the holder thread is
# wedged outside of ``iter_events`` and the in-process kill never fires.
# Forever-streaming sessions that keep producing data are intentionally
# never killed (#1709).
_DEFAULT_NO_REPLY_SECONDS: float = 3600.0


class SessionLockWatchdog:
    """Per-repo poller that evicts FSM lock holders past a no-reply
    deadline.

    Accepts *registry* and *repos* via the constructor so tests can
    drive it directly with hand-rolled fakes (no MagicMock per
    fido test conventions).  *no_reply_seconds* is configurable
    per instance: tests use a tiny value so a single iteration triggers
    eviction; production uses :data:`_DEFAULT_NO_REPLY_SECONDS`.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
        *,
        no_reply_seconds: float = _DEFAULT_NO_REPLY_SECONDS,
    ) -> None:
        self.registry = registry
        self.repos = repos
        self.no_reply_seconds = no_reply_seconds

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0.

        For each configured repo with an attached session, read
        :attr:`~fido.provider.OwnedSession.outstanding_send_at`.  If
        the clock is armed (not ``None``) and ``now -
        outstanding_send_at`` exceeds :attr:`no_reply_seconds`, fire
        :meth:`~fido.provider.OwnedSession.force_release`.

        * Idle sessions (clock is ``None``) are left alone forever.
        * Active turns that keep receiving data disarm the clock on
          every receive and so are left alone forever.
        * Only the **prompt-sent-no-reply** wedge — clock armed,
          no receive in too long — triggers a kill.

        The ``talker`` snapshot is consulted only to label the kill
        reason (which tid / kind / description is being evicted) — the
        kill decision itself depends only on the session's send/receive
        history.
        """
        for repo_name in self.repos:
            session = self._resolve_session(repo_name)
            if session is None:
                continue
            outstanding = session.outstanding_send_at
            if outstanding is None:
                continue
            silent_for = (talker_now() - outstanding).total_seconds()
            if silent_for <= self.no_reply_seconds:
                continue
            talker = get_talker(repo_name)
            label = (
                f"tid={talker.thread_id} kind={talker.kind}, "
                f"description={talker.description!r}"
                if talker is not None
                else "(no talker registered)"
            )
            reason = (
                f"no reply for {silent_for:.0f}s "
                f"(deadline {self.no_reply_seconds:.0f}s, holder {label})"
            )
            log.warning(
                "session-lock-watchdog[%s]: outstanding send past deadline — %s",
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
    no_reply_seconds: float = _DEFAULT_NO_REPLY_SECONDS,
) -> int:
    """Module-level entry point — create a watchdog and run one tick."""
    return SessionLockWatchdog(registry, repos, no_reply_seconds=no_reply_seconds).run()
