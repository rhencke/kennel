"""Gate test: fido status reflects live provider stats during an active turn.

Regression guard for the case where mid-turn snapshots showed zeros — the
snapshot publisher must fire immediately after each send/receive, not only
after the whole turn finishes.
"""

import json
import queue
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from frozendict import frozendict

from fido.appstate import (
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_PROVIDER,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderSnapshot,
    zero_repo_state,
)
from fido.atomic import AtomicUpdater, create_atomic
from fido.claude import ClaudeSession
from fido.provider import SnapshotPublisher
from fido.status import FidoStatus, RepoStatus, format_status

_REPO = "owner/repo"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _make_fido_state(repo_name: str) -> FidoState:
    """Return a minimal :class:`FidoState` with one pre-initialised repo entry."""
    return FidoState(
        repos=frozendict({repo_name: zero_repo_state(repo_name)}),
        github_limits=_ZERO_GITHUB_LIMITS,
        process_started_at=_EPOCH,
    )


class _StatePublisher:
    """Minimal :class:`~fido.provider.SnapshotPublisher` that writes into the
    atomic cell — mirrors what :meth:`~fido.session_agent.SessionBackedAgent.publish_metrics`
    does, without pulling in the full agent hierarchy.

    Also sets *received_event* once ``received_count > 0`` so the test's main
    thread can wait for the first mid-turn event without busy-polling.
    """

    def __init__(
        self,
        repo_name: str,
        updater: AtomicUpdater[FidoState],
        received_event: threading.Event,
    ) -> None:
        self._repo_name = repo_name
        self._updater = updater
        self._received_event = received_event

    def publish_metrics(
        self,
        *,
        owner: str | None,
        alive: bool,
        pid: int | None,
        dropped_count: int,
        sent_count: int,
        received_count: int,
    ) -> None:
        snapshot = ProviderSnapshot(
            session_owner=owner,
            session_alive=alive,
            session_pid=pid,
            session_dropped_count=dropped_count,
            session_sent_count=sent_count,
            session_received_count=received_count,
        )
        _name = self._repo_name
        self._updater.update(lambda root: root.repos[_name].provider, snapshot)
        if received_count > 0:
            self._received_event.set()


def _make_proc_with_queue(line_queue: "queue.Queue[str]") -> MagicMock:
    """Build a mock Popen whose ``readline`` blocks on *line_queue*."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 99999
    proc.stdin = MagicMock()
    proc.stdin.closed = False
    proc.stdout = MagicMock()
    proc.stdout.readline = line_queue.get  # blocks until a line is available
    proc.stderr = MagicMock()
    proc.poll = MagicMock(return_value=None)  # process is alive
    proc.wait = MagicMock(return_value=0)
    proc.returncode = 0
    return proc


def _make_queue_session(
    tmp_path: Path,
    line_queue: "queue.Queue[str]",
    publisher: SnapshotPublisher,
) -> ClaudeSession:
    """Construct a :class:`ClaudeSession` whose stdout is fed from *line_queue*.

    The injected *selector* always reports stdout as ready so the read loop
    never sleeps — the only actual blocking point is the ``readline()`` call
    itself, which pops from *line_queue*.
    """
    system_file = tmp_path / "system.md"
    system_file.write_text("you are fido")
    proc = _make_proc_with_queue(line_queue)
    fake_popen = MagicMock(return_value=proc)
    # selector always says proc.stdout is ready — no actual select() call
    fake_selector = MagicMock(return_value=([proc.stdout], [], []))
    return ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=fake_popen,
        selector=fake_selector,
        repo_name=_REPO,
        snapshot_publisher=publisher,
    )


class TestLiveProviderStats:
    def test_mid_turn_snapshot_shows_nonzero_counts(self, tmp_path: Path) -> None:
        """Snapshot must show sent/received > 0 while a turn is still in flight.

        This is the regression gate for the bug where mid-turn reads of the
        status snapshot returned zeros because the publisher only fired after
        the full turn completed.
        """
        initial = _make_fido_state(_REPO)
        reader, updater = create_atomic(initial)

        line_queue: queue.Queue[str] = queue.Queue()
        received_event = threading.Event()
        publisher = _StatePublisher(_REPO, updater, received_event)
        session = _make_queue_session(tmp_path, line_queue, publisher)

        errors: list[BaseException] = []

        def bg() -> None:
            try:
                session.send("do something")
                session.consume_until_result()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        bg_thread = threading.Thread(target=bg, daemon=True)
        bg_thread.start()

        # Feed one non-result event — this will unblock the first readline()
        # call inside iter_events() and bump received_count to 1, then the
        # background thread blocks again waiting for the next line.
        line_queue.put(json.dumps({"type": "assistant", "text": "working…"}) + "\n")

        # Wait until the snapshot shows received_count > 0 (the publisher
        # sets received_event immediately after the counter is incremented).
        assert received_event.wait(timeout=5), (
            "mid-turn snapshot never showed received_count > 0 — "
            "publisher fired too late or not at all"
        )

        # ── mid-turn snapshot assertions ──────────────────────────────────────
        snap = reader.get().repos[_REPO].provider
        assert snap is not None, "provider snapshot was not published mid-turn"
        assert snap.session_sent_count > 0, (
            f"expected sent_count > 0 mid-turn, got {snap.session_sent_count}"
        )
        assert snap.session_received_count > 0, (
            f"expected received_count > 0 mid-turn, got {snap.session_received_count}"
        )

        # ── primitive-type assertions: no mutable refs inside the snapshot ───
        assert isinstance(snap.session_sent_count, int)
        assert isinstance(snap.session_received_count, int)
        assert isinstance(snap.session_dropped_count, int)
        assert isinstance(snap.session_alive, bool)
        assert snap.session_pid is None or isinstance(snap.session_pid, int)
        assert snap.session_owner is None or isinstance(snap.session_owner, str)

        # Release the background thread: feed the result event so
        # consume_until_result() can return.
        line_queue.put(json.dumps({"type": "result", "result": "done"}) + "\n")
        bg_thread.join(timeout=5)
        assert not bg_thread.is_alive(), "background thread did not finish"
        assert not errors, f"background thread raised: {errors}"

    def test_no_publisher_leaves_snapshot_at_initial(self, tmp_path: Path) -> None:
        """Without a publisher, the atomic snapshot is never updated mid-turn.

        Negative-style gate: confirms the snapshot zero-state is not a
        coincidence of the publisher firing but the actual absence of wiring.
        """
        initial = _make_fido_state(_REPO)
        reader, _ = create_atomic(initial)

        line_queue: queue.Queue[str] = queue.Queue()
        system_file = tmp_path / "system.md"
        system_file.write_text("you are fido")
        proc = _make_proc_with_queue(line_queue)
        # No snapshot_publisher wired in — session publishes nothing.
        session = ClaudeSession(
            system_file,
            work_dir=tmp_path,
            popen=MagicMock(return_value=proc),
            selector=MagicMock(return_value=([proc.stdout], [], [])),
            repo_name=_REPO,
        )

        errors: list[BaseException] = []

        def bg() -> None:
            try:
                session.send("do something")
                session.consume_until_result()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        bg_thread = threading.Thread(target=bg, daemon=True)
        bg_thread.start()

        # Feed both events and let the turn complete.
        line_queue.put(json.dumps({"type": "assistant", "text": "working…"}) + "\n")
        line_queue.put(json.dumps({"type": "result", "result": "done"}) + "\n")
        bg_thread.join(timeout=5)
        assert not bg_thread.is_alive()
        assert not errors

        # The snapshot should never have been written — provider is still
        # the zero sentinel (per #1696, no None on appstate; missing data
        # is the zero ProviderSnapshot, not None).
        snap = reader.get().repos[_REPO].provider
        assert snap == _ZERO_PROVIDER, (
            f"expected provider snapshot to remain _ZERO_PROVIDER without a publisher, got {snap}"
        )

    def test_display_path_shows_counter_values(self) -> None:
        """Counter values from a ProviderSnapshot appear in format_status output.

        Loose assertion — does not pin exact formatting, only that the integers
        are present somewhere in the rendered string.
        """
        sent = 7
        received = 14
        repo = RepoStatus(
            name=_REPO,
            fido_running=True,
            issue=None,
            pending=0,
            completed=0,
            current_task=None,
            claude_pid=None,
            claude_uptime=None,
            worker_what="coding",
            crash_count=0,
            last_crash_error=None,
            worker_stuck=False,
            session_alive=True,  # required for _format_agent_line to render stats
            session_sent_count=sent,
            session_received_count=received,
        )
        status = FidoStatus(fido_pid=None, fido_uptime=None, repos=[repo])
        rendered = format_status(status)
        assert str(sent) in rendered, (
            f"session_sent_count {sent} not found in rendered status output"
        )
        assert str(received) in rendered, (
            f"session_received_count {received} not found in rendered status output"
        )
