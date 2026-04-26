"""Regression tests for the watchdog_transitions FSM oracle.

Each test section corresponds to a proved invariant from
``models/watchdog_transitions.v``.  The tests exercise the extracted
``transition`` function directly and verify that ``Watchdog._fsm_transition``
raises ``AssertionError`` on any rejected transition — enforcing the
fail-closed contract.

Proved invariants exercised:

  ``alive_never_forcibly_restarted`` — WatchdogDetectDead rejected from
    Running and Hung; the dead-detection path requires first passing
    through Crashed via WorkerCrash.

  ``hung_not_restarted``            — WatchdogDetectAlive from Hung stays
    Hung; a stale-but-alive thread is never spontaneously restarted.

  ``dead_always_restarted``         — WatchdogDetectDead from Crashed is
    always accepted (Some Restarting); restart is total from Crashed.

  ``restart_reaches_running``       — RestartComplete from Restarting
    always yields Running; a successful restart is not partial.

  ``stopped_is_terminal``           — every event from Stopped is rejected;
    orderly shutdown cannot be undone by the watchdog.

Field lesson covered:

  Forced restart of a live thread raced on the fido lockfile (the old
  thread might not exit before the new one starts, and Python threads
  can't be cleanly killed), causing a restart loop in production.
  The ``alive_never_forcibly_restarted`` invariant machine-checks that
  this class of bug cannot recur: WatchdogDetectDead is always rejected
  from Running and Hung.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido.config import RepoConfig as _RepoConfig
from fido.provider import ProviderID
from fido.rocq.watchdog_transitions import (
    ActivityResume,
    Crashed,
    Hung,
    RestartComplete,
    RestartCrash,
    Restarting,
    Running,
    StaleTimeout,
    State,
    Stopped,
    StopRequest,
    WatchdogDetectAlive,
    WatchdogDetectDead,
    WorkerCrash,
    transition,
)
from fido.watchdog import Watchdog


class RepoConfig(_RepoConfig):
    def __init__(self, *args, provider: ProviderID = ProviderID.CLAUDE_CODE, **kwargs):
        super().__init__(*args, provider=provider, **kwargs)


def _repo(name: str = "owner/repo") -> RepoConfig:
    return RepoConfig(name=name, work_dir=Path("/tmp/repo"))


def _watchdog(
    repos: dict[str, RepoConfig] | None = None,
) -> tuple[Watchdog, MagicMock]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = MagicMock()
    return Watchdog(registry, repos), registry


# ---------------------------------------------------------------------------
# Invariant: alive_never_forcibly_restarted
#
# Field lesson: forced restart of a live thread raced on the fido lockfile.
# The old thread may not exit before the new one starts; Python threads
# can't be cleanly killed.  This caused a restart loop in production.
# The FSM encodes the rule: WatchdogDetectDead is only valid from Crashed,
# never from any alive state (Running or Hung).
# ---------------------------------------------------------------------------


def test_alive_never_forcibly_restarted_from_running() -> None:
    """WatchdogDetectDead is rejected from Running — alive thread not restarted.

    Pre-fix scenario: the watchdog fired a restart while is_alive() returned
    True, entering a restart loop.  The FSM rejects WatchdogDetectDead from
    Running, machine-checking this constraint.
    """
    result = transition(Running(), WatchdogDetectDead())
    assert result is None, (
        "alive_never_forcibly_restarted violated: "
        "WatchdogDetectDead accepted from Running"
    )


def test_alive_never_forcibly_restarted_from_hung() -> None:
    """WatchdogDetectDead is rejected from Hung — stale thread not restarted.

    A thread alive but beyond the stale threshold is display-only Hung.
    WatchdogDetectDead from Hung is rejected: alive threads — even stale
    ones — must never be forcibly restarted by the watchdog.
    """
    result = transition(Hung(), WatchdogDetectDead())
    assert result is None, (
        "alive_never_forcibly_restarted violated: WatchdogDetectDead accepted from Hung"
    )


def test_detect_dead_only_valid_from_crashed() -> None:
    """WatchdogDetectDead is accepted only from Crashed — the sole restart entry.

    The only path to Restarting is through Crashed.  This means the watchdog
    must always observe a WorkerCrash (dead thread) before initiating restart —
    it cannot skip the Crashed state.
    """
    assert transition(Crashed(), WatchdogDetectDead()) == Restarting()
    assert transition(Running(), WatchdogDetectDead()) is None
    assert transition(Hung(), WatchdogDetectDead()) is None
    assert transition(Restarting(), WatchdogDetectDead()) is None
    assert transition(Stopped(), WatchdogDetectDead()) is None


# ---------------------------------------------------------------------------
# Invariant: hung_not_restarted
#
# WatchdogDetectAlive from Hung stays Hung.  A stale-but-alive thread is
# left to time out on its own (claude has its own idle timeout) or recover
# via ActivityResume.  The watchdog observes but does not restart.
# ---------------------------------------------------------------------------


def test_hung_not_restarted_stays_hung() -> None:
    """WatchdogDetectAlive from Hung stays Hung — no spontaneous restart.

    The hung_not_restarted invariant: a watchdog tick that observes an
    alive-but-stale thread must not trigger a restart.  The state remains
    Hung until the thread either recovers or crashes on its own.
    """
    result = transition(Hung(), WatchdogDetectAlive())
    assert isinstance(result, Hung), (
        "hung_not_restarted violated: WatchdogDetectAlive from Hung "
        f"yielded {type(result).__name__!r} instead of Hung"
    )
    assert not isinstance(result, Running)
    assert not isinstance(result, Crashed)


def test_hung_recovers_to_running_via_activity_resume() -> None:
    """ActivityResume from Hung returns to Running — self-recovery path.

    A hung thread that starts reporting activity again transitions to
    Running without the watchdog having to restart it.
    """
    result = transition(Hung(), ActivityResume())
    assert isinstance(result, Running)


def test_hung_can_crash() -> None:
    """WorkerCrash from Hung reaches Crashed — crash detectable from any alive state.

    Even a stale thread can crash.  Hung → WorkerCrash → Crashed ensures
    the subsequent WatchdogDetectDead transition is still valid.
    """
    result = transition(Hung(), WorkerCrash())
    assert isinstance(result, Crashed)


def test_running_enters_hung_via_stale_timeout() -> None:
    """StaleTimeout from Running reaches Hung — stale detection path.

    When a running thread exceeds the stale threshold, the FSM enters Hung.
    From Hung the watchdog observes but does not restart.
    """
    result = transition(Running(), StaleTimeout())
    assert isinstance(result, Hung)


# ---------------------------------------------------------------------------
# Invariant: dead_always_restarted
#
# WatchdogDetectDead from Crashed always yields Restarting.  Restart is a
# total function from Crashed — the watchdog never silently ignores a dead
# thread.
# ---------------------------------------------------------------------------


def test_dead_always_restarted_from_crashed() -> None:
    """WatchdogDetectDead from Crashed yields Restarting — restart is total.

    dead_always_restarted: every crash the watchdog detects becomes a
    restart attempt.  There is no conditional path where a dead thread
    stays dead without the watchdog initiating Restarting.
    """
    result = transition(Crashed(), WatchdogDetectDead())
    assert isinstance(result, Restarting), (
        "dead_always_restarted violated: WatchdogDetectDead from Crashed "
        f"yielded {type(result).__name__!r} instead of Restarting"
    )


def test_full_crash_restart_path() -> None:
    """Running → WorkerCrash → Crashed → WatchdogDetectDead → Restarting → RestartComplete → Running.

    The complete crash/restart lifecycle succeeds at every step and
    returns the thread to Running with no stuck intermediate states.
    """
    s: State = Running()

    s = transition(s, WorkerCrash())  # type: ignore[assignment]
    assert isinstance(s, Crashed)

    s = transition(s, WatchdogDetectDead())  # type: ignore[assignment]
    assert isinstance(s, Restarting)

    s = transition(s, RestartComplete())  # type: ignore[assignment]
    assert isinstance(s, Running)


def test_repeated_crash_restart_cycle() -> None:
    """Thread can crash and restart multiple times — no FSM exhaustion.

    Each cycle goes Running → WorkerCrash → Crashed → WatchdogDetectDead
    → Restarting → RestartComplete → Running.  The FSM supports arbitrarily
    many crash/restart cycles.
    """
    s: State = Running()
    for _ in range(3):
        s = transition(s, WorkerCrash())  # type: ignore[assignment]
        assert isinstance(s, Crashed)
        s = transition(s, WatchdogDetectDead())  # type: ignore[assignment]
        assert isinstance(s, Restarting)
        s = transition(s, RestartComplete())  # type: ignore[assignment]
        assert isinstance(s, Running)


# ---------------------------------------------------------------------------
# Invariant: restart_reaches_running
#
# RestartComplete from Restarting always yields Running.  A successful
# restart is not partial: the new thread enters Running with its rescued
# provider in hand.
# ---------------------------------------------------------------------------


def test_restart_reaches_running() -> None:
    """RestartComplete from Restarting yields Running — restart is complete.

    restart_reaches_running: the new thread is alive and has the rescued
    provider.  The FSM enters Running and subsequent watchdog ticks will
    see WatchdogDetectAlive → Running.
    """
    result = transition(Restarting(), RestartComplete())
    assert isinstance(result, Running), (
        "restart_reaches_running violated: RestartComplete from Restarting "
        f"yielded {type(result).__name__!r} instead of Running"
    )


def test_restart_crash_escalation_path() -> None:
    """RestartCrash from Restarting returns to Crashed — escalation modelled.

    The replacement thread can also crash immediately.  RestartCrash → Crashed
    means the next watchdog tick detects the dead thread and attempts another
    restart.  Escalation is a cycle (Crashed → Restarting → Crashed) rather
    than a stuck state.
    """
    result = transition(Restarting(), RestartCrash())
    assert isinstance(result, Crashed)


# ---------------------------------------------------------------------------
# Invariant: stopped_is_terminal
#
# Every event from Stopped is rejected.  Orderly shutdown is terminal:
# the watchdog must not restart a thread that was deliberately stopped.
# ---------------------------------------------------------------------------


def test_running_to_stopped_via_stop_request() -> None:
    """StopRequest from Running reaches Stopped — orderly shutdown path."""
    result = transition(Running(), StopRequest())
    assert isinstance(result, Stopped)


def test_hung_to_stopped_via_stop_request() -> None:
    """StopRequest from Hung reaches Stopped — stale thread cleanly stopped."""
    result = transition(Hung(), StopRequest())
    assert isinstance(result, Stopped)


def test_stopped_is_terminal_complete() -> None:
    """All eight events are rejected from Stopped — orderly shutdown is terminal.

    stopped_is_terminal: no event can restart or transition out of Stopped.
    The watchdog's supervision contract ends the moment stop() is called.
    """
    s = Stopped()
    for event in [
        WorkerCrash(),
        WatchdogDetectAlive(),
        WatchdogDetectDead(),
        StaleTimeout(),
        ActivityResume(),
        RestartComplete(),
        RestartCrash(),
        StopRequest(),
    ]:
        result = transition(s, event)
        assert result is None, (
            f"stopped_is_terminal violated: {type(event).__name__} accepted from Stopped"
        )


# ---------------------------------------------------------------------------
# Watchdog._fsm_transition crash-loud behaviour
#
# Verify that Watchdog._fsm_transition raises AssertionError on invalid
# events, not silently no-ops.  This is the fail-closed contract: a
# coordination violation surfaces as an immediate crash rather than silent
# state divergence.
# ---------------------------------------------------------------------------


def test_fsm_transition_crashes_on_invalid_event() -> None:
    """_fsm_transition raises AssertionError when the FSM rejects an event.

    Starts the repo in its initial Running state and fires WatchdogDetectDead —
    which the FSM rejects (Running + WatchdogDetectDead → None).  The oracle
    must raise AssertionError immediately, enforcing alive_never_forcibly_restarted
    at runtime.
    """
    w, _registry = _watchdog()
    # Default initial state is Running; WatchdogDetectDead is rejected from Running.
    with pytest.raises(AssertionError, match="watchdog_transitions FSM"):
        w._fsm_transition("owner/repo", WatchdogDetectDead())  # pyright: ignore[reportPrivateUsage]


def test_fsm_transition_includes_repo_name_in_error() -> None:
    """AssertionError message names the repo so the violation is easy to locate."""
    w, _registry = _watchdog({"myorg/myrepo": _repo("myorg/myrepo")})
    with pytest.raises(AssertionError, match="myorg/myrepo"):
        w._fsm_transition("myorg/myrepo", WatchdogDetectDead())  # pyright: ignore[reportPrivateUsage]


def test_fsm_transition_includes_state_and_event_in_error() -> None:
    """AssertionError message names the rejected state and event."""
    w, _registry = _watchdog()
    with pytest.raises(AssertionError, match="Running") as exc_info:
        w._fsm_transition("owner/repo", WatchdogDetectDead())  # pyright: ignore[reportPrivateUsage]
    assert "WatchdogDetectDead" in str(exc_info.value)


def test_fsm_transition_initialises_state_to_running() -> None:
    """_fsm_transition initialises a new repo's state to Running on first access.

    The first successful transition from an unknown repo uses Running as the
    implicit starting state — matching the lifecycle expectation that a thread
    starts alive.
    """
    w, _registry = _watchdog()
    new_state = w._fsm_transition("owner/repo", WatchdogDetectAlive())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(new_state, Running)
    assert isinstance(w._fsm_states.get("owner/repo"), Running)  # pyright: ignore[reportPrivateUsage]


def test_fsm_transition_persists_state_between_calls() -> None:
    """_fsm_transition updates _fsm_states; subsequent calls see the new state."""
    w, _registry = _watchdog()
    # Running → WorkerCrash → Crashed
    w._fsm_transition("owner/repo", WorkerCrash())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(w._fsm_states.get("owner/repo"), Crashed)  # pyright: ignore[reportPrivateUsage]
    # Crashed → WatchdogDetectDead → Restarting
    w._fsm_transition("owner/repo", WatchdogDetectDead())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(w._fsm_states.get("owner/repo"), Restarting)  # pyright: ignore[reportPrivateUsage]
    # Restarting → RestartComplete → Running
    w._fsm_transition("owner/repo", RestartComplete())  # pyright: ignore[reportPrivateUsage]
    assert isinstance(w._fsm_states.get("owner/repo"), Running)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Oracle integration: Watchdog.run drives the FSM state machine
# ---------------------------------------------------------------------------


def test_run_advances_fsm_on_alive_tick() -> None:
    """Watchdog.run fires WatchdogDetectAlive and leaves FSM in Running."""
    w, registry = _watchdog()
    registry.is_alive.return_value = True
    w.run()
    assert isinstance(w._fsm_states.get("owner/repo"), Running)  # pyright: ignore[reportPrivateUsage]


def test_run_advances_fsm_through_crash_restart_cycle() -> None:
    """Watchdog.run drives the full crash/restart lifecycle through the oracle.

    Dead-thread tick: Running → WorkerCrash → Crashed → WatchdogDetectDead
    → Restarting → registry.start → RestartComplete → Running.

    Alive-thread tick: Running → WatchdogDetectAlive → Running.

    The FSM ends in Running after both ticks, confirming the restart
    completed and the thread is back to normal.
    """
    repo_cfg = _repo()
    w, registry = _watchdog({"owner/repo": repo_cfg})

    # Tick 1: dead thread — full crash/restart path.
    registry.is_alive.return_value = False
    registry.get_thread_crash_error.return_value = None
    w.run()
    assert isinstance(w._fsm_states.get("owner/repo"), Running)  # pyright: ignore[reportPrivateUsage]

    # Tick 2: alive thread — WatchdogDetectAlive stays Running.
    registry.is_alive.return_value = True
    w.run()
    assert isinstance(w._fsm_states.get("owner/repo"), Running)  # pyright: ignore[reportPrivateUsage]


def test_run_tracks_fsm_independently_per_repo() -> None:
    """FSM state is tracked independently for each repo.

    Two repos with different liveness: alive stays Running, dead goes
    through crash/restart.  No cross-repo contamination.
    """
    alive_cfg = _repo("org/alive")
    dead_cfg = _repo("org/dead")
    repos = {"org/alive": alive_cfg, "org/dead": dead_cfg}
    w, registry = _watchdog(repos)

    def is_alive(name: str) -> bool:
        return name == "org/alive"

    registry.is_alive.side_effect = is_alive
    registry.get_thread_crash_error.return_value = None
    w.run()

    assert isinstance(w._fsm_states.get("org/alive"), Running)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(w._fsm_states.get("org/dead"), Running)  # pyright: ignore[reportPrivateUsage]  # after restart
