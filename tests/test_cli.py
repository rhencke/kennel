"""Tests for fido.cli — add/complete/list subcommands."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fido.cli import Cmd, build_parser, main
from fido.types import TaskType

# ── helpers ───────────────────────────────────────────────────────────────────


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


# ── build_parser ──────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_add_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "spec", "my task"])
        assert args.command == "add"
        assert args.task_type == TaskType.SPEC
        assert args.title == "my task"
        assert args.description == ""

    def test_add_with_description(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "ci", "title", "desc"])
        assert args.task_type == TaskType.CI
        assert args.description == "desc"

    def test_add_with_comment_id(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                str(tmp_path),
                "add",
                "thread",
                "my task",
                "--comment-id",
                "42",
                "--repo",
                "a/b",
                "--pr",
                "7",
            ]
        )
        assert args.task_type == TaskType.THREAD
        assert args.comment_id == 42
        assert args.repo == "a/b"
        assert args.pr == 7

    def test_add_without_comment_id_defaults_none(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "spec", "my task"])
        assert args.comment_id is None
        assert args.repo is None
        assert args.pr is None

    def test_complete_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "complete", "task-id-123"])
        assert args.command == "complete"
        assert args.task_id == "task-id-123"

    def test_list_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "list"])
        assert args.command == "list"

    def test_no_command_exits(self, tmp_path: Path) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([str(tmp_path)])


# ── Cmd.add ───────────────────────────────────────────────────────────────────


class TestCmdAdd:
    def test_adds_task(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(
            tmp_path, TaskType.SPEC, "my task", "some description"
        )
        capsys.readouterr()  # consume add output
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "my task"
        assert tasks[0]["description"] == "some description"
        assert tasks[0]["type"] == "spec"

    def test_adds_task_no_description(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(tmp_path, TaskType.CI, "bare task", "")
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["description"] == ""
        assert tasks[0]["type"] == "ci"

    def test_adds_task_with_comment_id_builds_thread(
        self, tmp_path: Path, capsys
    ) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(
            tmp_path, TaskType.THREAD, "threaded", "", comment_id=42, repo="a/b", pr=7
        )
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["thread"] == {"comment_id": 42, "repo": "a/b", "pr": 7}

    def test_adds_task_comment_id_only(self, tmp_path: Path, capsys) -> None:
        """comment_id without repo/pr still sets a thread for dedup purposes."""
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(
            tmp_path, TaskType.THREAD, "threaded", "", comment_id=99
        )
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["thread"] == {"comment_id": 99}

    def test_add_deduplicates_by_comment_id(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(
            tmp_path,
            TaskType.THREAD,
            "first title",
            "",
            comment_id=42,
            repo="a/b",
            pr=7,
        )
        capsys.readouterr()
        Cmd(github=MagicMock()).add(
            tmp_path,
            TaskType.THREAD,
            "different title",
            "",
            comment_id=42,
            repo="a/b",
            pr=7,
        )
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "first title"

    def test_add_prints_task_json(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).add(tmp_path, TaskType.SPEC, "my task", "")
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["title"] == "my task"
        assert "id" in data


# ── Cmd.complete ──────────────────────────────────────────────────────────────


class TestCmdComplete:
    def test_completes_task_no_thread(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        cmd = Cmd(github=MagicMock())
        task = cmd.add(tmp_path, TaskType.SPEC, "task to finish", "")
        capsys.readouterr()
        cmd.complete(tmp_path, task["id"])
        from fido.tasks import list_tasks

        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_completes_task_with_thread_resolves(
        self, tmp_path: Path, capsys, caplog
    ) -> None:
        _task_file(tmp_path)
        from fido.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(
            tmp_path, title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = MagicMock()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "reviewer"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
        # The auto-resolve oracle reads thread.comments.nodes (not the
        # flat get_pull_comments list) to decide who was last in the
        # thread.  Include fido's reply (id=99) so the decision is
        # ResolveReviewThread (we posted last).
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "reviewer"}},
                        {"databaseId": 99, "author": {"login": "fido-bot"}},
                    ],
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Cmd(github=mock_github).complete(tmp_path, task["id"])

        mock_github.resolve_thread.assert_called_once_with("thread_node_abc")
        assert "thread resolved: thread_node_abc" in caplog.text

    def test_completes_task_with_thread_skips_if_not_last(
        self, tmp_path: Path, capsys, caplog
    ) -> None:
        _task_file(tmp_path)
        from fido.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(
            tmp_path, title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = MagicMock()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "copilot[bot]"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
        # Bot reviewer was last in the thread.  ``[bot]``-suffixed authors
        # are classified as ``CommentByBot`` by the auto-resolve oracle,
        # which excludes them from being "last fido comment", so the
        # oracle decides DON'T resolve.  (Plain "reviewer" without bot
        # suffix or collaborator membership is classified as
        # ``CommentIgnored`` and skipped, which would let fido be
        # considered last — that's a genuine CLI gap when collaborator
        # metadata isn't available, tested elsewhere.)
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "fido-bot"}},
                        {"databaseId": 99, "author": {"login": "copilot[bot]"}},
                    ],
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            Cmd(github=mock_github).complete(tmp_path, task["id"])

        mock_github.resolve_thread.assert_not_called()
        assert "not resolving" in caplog.text

    def test_completes_task_with_thread_no_matching_comments(
        self, tmp_path: Path
    ) -> None:
        _task_file(tmp_path)
        from fido.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(
            tmp_path, title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = MagicMock()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = []

        Cmd(github=mock_github).complete(tmp_path, task["id"])

        mock_github.resolve_thread.assert_not_called()

    def test_completes_task_with_thread_exception_silenced(
        self, tmp_path: Path, caplog
    ) -> None:
        _task_file(tmp_path)
        from fido.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(
            tmp_path, title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = MagicMock()
        mock_github.get_user.side_effect = RuntimeError("network error")

        # Should not raise; exception is swallowed and logged
        with caplog.at_level(logging.WARNING, logger="fido"):
            Cmd(github=mock_github).complete(tmp_path, task["id"])
        assert "thread resolution skipped" in caplog.text

    def test_completes_task_with_thread_already_resolved(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        from fido.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = add_task(
            tmp_path, title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = MagicMock()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
        ]
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": True,
                "comments": {"nodes": [{"databaseId": 42}]},
            }
        ]

        Cmd(github=mock_github).complete(tmp_path, task["id"])

        mock_github.resolve_thread.assert_not_called()

    def test_thread_missing_fields_skips(self, tmp_path: Path) -> None:
        """Thread dict with missing fields should silently skip resolution."""
        _task_file(tmp_path)
        from fido.tasks import add_task

        # thread missing 'pr' and 'comment_id'
        task = add_task(
            tmp_path, title="task", task_type=TaskType.THREAD, thread={"repo": "a/b"}
        )

        mock_github = MagicMock()
        Cmd(github=mock_github).complete(tmp_path, task["id"])

        mock_github.resolve_thread.assert_not_called()

    def test_complete_nonexistent_id_no_error(self, tmp_path: Path) -> None:
        """Completing a non-existent task ID should not raise."""
        _task_file(tmp_path)
        Cmd(github=MagicMock()).complete(tmp_path, "nonexistent-id")


# ── Cmd.list ──────────────────────────────────────────────────────────────────


class TestCmdList:
    def test_prints_json(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        cmd = Cmd(github=MagicMock())
        cmd.add(tmp_path, TaskType.SPEC, "alpha", "")
        capsys.readouterr()
        cmd.add(tmp_path, TaskType.SPEC, "beta", "desc")
        capsys.readouterr()
        cmd.list(tmp_path)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2
        assert data[0]["title"] == "alpha"
        assert data[1]["title"] == "beta"

    def test_empty_list(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd(github=MagicMock()).list(tmp_path)
        out = capsys.readouterr().out
        assert json.loads(out) == []


# ── main (integration) ────────────────────────────────────────────────────────


class TestMain:
    def test_add_via_main(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "task title"], _GitHub=MagicMock)
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "task title"

    def test_add_via_main_with_comment_id(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        main(
            [
                str(tmp_path),
                "add",
                "thread",
                "task title",
                "--comment-id",
                "55",
                "--repo",
                "r/r",
                "--pr",
                "3",
            ],
            _GitHub=MagicMock,
        )
        capsys.readouterr()
        from fido.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["thread"] == {"comment_id": 55, "repo": "r/r", "pr": 3}

    def test_complete_via_main(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "finish me"], _GitHub=MagicMock)
        out = capsys.readouterr().out
        task_id = json.loads(out)["id"]
        main([str(tmp_path), "complete", task_id], _GitHub=MagicMock)
        from fido.tasks import list_tasks

        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_list_via_main(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "one"], _GitHub=MagicMock)
        capsys.readouterr()
        main([str(tmp_path), "list"], _GitHub=MagicMock)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["title"] == "one"

    def test_no_args_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_raises(self, tmp_path: Path, monkeypatch) -> None:
        """Fallback case in match statement raises AssertionError."""
        from unittest.mock import MagicMock

        import fido.cli as cli_mod

        fake_args = MagicMock()
        fake_args.command = "bogus"
        fake_parser = MagicMock()
        fake_parser.parse_args.return_value = fake_args

        monkeypatch.setattr(cli_mod, "build_parser", lambda: fake_parser)
        with pytest.raises(AssertionError, match="unreachable"):
            main([], _GitHub=MagicMock)
