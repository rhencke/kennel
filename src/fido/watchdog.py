"""Watchdog — check WorkerThread health and restart dead threads."""

import logging
import subprocess
import threading
import time
from datetime import datetime, timezone

import requests

from fido.config import RepoConfig
from fido.github import GitHub, GraphQLError
from fido.registry import WorkerRegistry
from fido.rocq import watchdog_transitions as watchdog_fsm

log = logging.getLogger(__name__)

_WATCHDOG_INTERVAL: float = 30.0
# Display-only: /status endpoint flags workers with no activity for this
# long as "stuck" in the UI.  Not used for restart decisions (see class
# docstring).
_STALE_THRESHOLD: float = 600.0
# Hourly reconcile cadence (closes #812) — webhooks keep the cache fresh
# in steady state; this catches drift from any lost events.
_RECONCILE_INTERVAL: float = 3600.0


class Watchdog:
    """Check WorkerThread health and restart dead threads.

    Accepts *registry* and *repos* via the constructor so tests can inject
    mock objects without patching module-level names.

    Dead threads (is_alive returns False) are restarted immediately.

    Stale threads (alive but no activity reported) are left alone.  The
    claude subprocess has its own idle timeout, so a worker that looks
    stuck is either waiting for claude to finish or will bubble up a
    timeout error on its own.  A forced restart of a live thread races on
    the fido lockfile (the old thread may not exit before the new one
    starts, and Python threads can't be cleanly killed), which caused a
    restart loop in the field — so we no longer do it.

    :attr:`_fsm_states` tracks the formal lifecycle state for each repo
    via the FSM extracted from ``models/watchdog_transitions.v``.  Every
    state change is asserted against the formal transition table so
    coordination bugs surface as crashes rather than silent drift.
    The two invariants this oracle enforces on every tick:

    - **alive_never_forcibly_restarted** — an alive thread never receives
      ``WatchdogDetectDead``; the dead-detection path only fires when
      ``is_alive`` returns ``False``.
    - **dead_always_restarted** — when dead is detected, ``WatchdogDetectDead``
      from ``Crashed`` always succeeds; a rejected transition would crash
      the watchdog, surfacing the coordination violation immediately.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
    ) -> None:
        self.registry = registry
        self.repos = repos
        # Per-repo FSM state from watchdog_transitions.v.  Only accessed
        # from the single watchdog daemon thread, so no lock is needed.
        self._fsm_states: dict[str, watchdog_fsm.State] = {}

    def _fsm_transition(
        self, repo_name: str, event: watchdog_fsm.Event
    ) -> watchdog_fsm.State:
        """Fire *event* for *repo_name*, raising ``AssertionError`` if rejected.

        Single oracle for every FSM transition in :meth:`run`, so a
        coordination bug surfaces as a crash rather than silent drift.
        Initialises the repo's FSM state to ``Running`` on first access.
        """
        prev = self._fsm_states.get(repo_name, watchdog_fsm.Running())
        new_state = watchdog_fsm.transition(prev, event)
        if new_state is None:
            raise AssertionError(
                f"watchdog_transitions FSM: {type(event).__name__} rejected in "
                f"state {type(prev).__name__} for repo {repo_name!r}"
            )
        self._fsm_states[repo_name] = new_state
        log.debug(
            "watchdog[%s]: FSM %s →%s via %s",
            repo_name,
            type(prev).__name__,
            type(new_state).__name__,
            type(event).__name__,
        )
        return new_state

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0.

        Current design — oracle mode: the hand-written ``if is_alive`` branch
        drives the restart logic; ``_fsm_transition`` fires alongside each
        decision as a crash-loud assertion.  A rejected FSM transition means the
        code diverged from the formal model and surfaces immediately.

        **E1 flip point**: when the E-band work lands, this method can be
        rewritten as a thin driver that feeds one of ``{WatchdogDetectAlive,
        WatchdogDetectDead}`` into the extracted ``watchdog_fsm.transition``
        and dispatches on the returned state — replacing the hand-written
        ``if not is_alive → start`` conditional entirely.  The oracle assertions
        become the control flow; the formal model drives the code rather than
        checking it from the side.  See ``models/watchdog_transitions.v`` for
        the full flip-point note.
        """
        for repo_name, repo_cfg in self.repos.items():
            if self.registry.is_alive(repo_name):
                # Thread alive: assert the FSM accepts the alive-observation
                # event.  This enforces alive_never_forcibly_restarted —
                # WatchdogDetectDead is only valid from Crashed, never from
                # Running or Hung.
                self._fsm_transition(repo_name, watchdog_fsm.WatchdogDetectAlive())
            else:
                error = self.registry.get_thread_crash_error(repo_name)
                if error is not None:
                    self.registry.record_crash(repo_name, error)
                log.info("thread for %s is not alive — restarting", repo_name)
                # Running/Hung → Crashed: thread exited unexpectedly.
                self._fsm_transition(repo_name, watchdog_fsm.WorkerCrash())
                # Crashed → Restarting: watchdog initiates the restart.
                self._fsm_transition(repo_name, watchdog_fsm.WatchdogDetectDead())
                self.registry.start(repo_cfg)
                # Restarting → Running: new thread started, provider rescued.
                self._fsm_transition(repo_name, watchdog_fsm.RestartComplete())
        return 0

    def start_thread(
        self, *, _interval: float = _WATCHDOG_INTERVAL
    ) -> threading.Thread:
        """Start a single daemon thread that periodically checks all repos."""

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.run()

        t = threading.Thread(target=_loop, daemon=True, name="watchdog")
        t.start()
        return t


def run(registry: WorkerRegistry, repos: dict[str, RepoConfig]) -> int:
    """Module-level entry point: create a Watchdog and run it."""
    return Watchdog(registry, repos).run()


class ReconcileWatchdog:
    """Hourly cache reconcile to heal drift from missed webhooks (#812).

    For each repo, fetches a fresh ``find_all_open_issues`` snapshot and
    calls :meth:`~fido.issue_cache.IssueTreeCache.reconcile_with_inventory`
    to apply any divergence.  Skips repos whose cache hasn't been
    bootstrapped yet — the worker thread does that on its first
    ``find_next_issue`` iteration.

    A failed ``find_all_open_issues`` due to a transient GitHub/network
    error logs the exception and continues to the next repo — the next
    hourly tick will retry.  Logic bugs propagate and crash the thread
    loudly so they aren't silently swallowed by the last safety net.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
        gh: GitHub,
    ) -> None:
        self.registry = registry
        self.repos = repos
        self.gh = gh

    def run(self) -> int:
        """Run one reconcile pass across every repo with a loaded cache.

        Returns 0 (parallels :class:`Watchdog.run` for symmetry).
        """
        for repo_name in self.repos:
            cache = self.registry.get_issue_cache(repo_name)
            if not cache.is_loaded:
                log.info(
                    "reconcile-watchdog[%s]: cache not yet loaded — skipping",
                    repo_name,
                )
                continue
            owner, name = repo_name.split("/", 1)
            snapshot_started_at = datetime.now(tz=timezone.utc)
            try:
                inventory = self.gh.find_all_open_issues(owner, name)
            except (
                requests.RequestException,
                GraphQLError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                # Transient GitHub/network failure — next hourly tick retries.
                # Logic bugs (KeyError, TypeError, AttributeError, …) are NOT
                # caught; they propagate and crash this thread loudly.
                log.exception(
                    "reconcile-watchdog[%s]: find_all_open_issues failed — "
                    "next hourly tick will retry",
                    repo_name,
                )
                continue
            drift = cache.reconcile_with_inventory(
                inventory, snapshot_started_at=snapshot_started_at
            )
            log.info("reconcile-watchdog[%s]: applied %d corrections", repo_name, drift)
        return 0

    def start_thread(
        self, *, _interval: float = _RECONCILE_INTERVAL
    ) -> threading.Thread:
        """Start a single daemon thread that reconciles every *_interval* seconds."""

        def _loop() -> None:
            while True:
                time.sleep(_interval)
                self.run()

        t = threading.Thread(target=_loop, daemon=True, name="reconcile-watchdog")
        t.start()
        return t
