"""Tests for kennel.cli — add/complete/list subcommands."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.cli import Cmd, build_parser, main

# ── helpers ───────────────────────────────────────────────────────────────────


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


# ── build_parser ──────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_add_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "my task"])
        assert args.command == "add"
        assert args.title == "my task"
        assert args.description == ""

    def test_add_with_description(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "title", "desc"])
        assert args.description == "desc"

    def test_complete_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "complete", "my task"])
        assert args.command == "complete"
        assert args.title == "my task"

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
    def test_adds_task(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        Cmd().add(tmp_path, "my task", "some description")
        from kennel.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert len(tasks) == 1
        assert tasks[0]["title"] == "my task"
        assert tasks[0]["description"] == "some description"

    def test_adds_task_no_description(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        Cmd().add(tmp_path, "bare task", "")
        from kennel.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["description"] == ""


# ── Cmd.complete ──────────────────────────────────────────────────────────────


class TestCmdComplete:
    def test_completes_task_no_thread(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        cmd = Cmd()
        cmd.add(tmp_path, "task to finish", "")
        cmd.complete(tmp_path, "task to finish")
        from kennel.tasks import list_tasks

        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_completes_task_with_thread_resolves(self, tmp_path: Path, caplog) -> None:
        _task_file(tmp_path)
        from kennel.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        add_task(tmp_path, title="threaded task", thread=thread)

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
        mock_github.get_review_threads.return_value = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [
                                {
                                    "id": "thread_node_abc",
                                    "isResolved": False,
                                    "comments": {"nodes": [{"databaseId": 42}]},
                                }
                            ]
                        }
                    }
                }
            }
        }

        with caplog.at_level(logging.INFO, logger="kennel"):
            Cmd(github=mock_github).complete(tmp_path, "threaded task")

        mock_github.resolve_thread.assert_called_once_with("thread_node_abc")
        assert "thread resolved: thread_node_abc" in caplog.text

    def test_completes_task_with_thread_skips_if_not_last(
        self, tmp_path: Path, caplog
    ) -> None:
        _task_file(tmp_path)
        from kennel.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        add_task(tmp_path, title="threaded task", thread=thread)

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
                "user": {"login": "reviewer"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]

        with caplog.at_level(logging.INFO, logger="kennel"):
            Cmd(github=mock_github).complete(tmp_path, "threaded task")

        mock_github.resolve_thread.assert_not_called()
        assert "not resolving" in caplog.text

    def test_completes_task_with_thread_no_matching_comments(
        self, tmp_path: Path
    ) -> None:
        _task_file(tmp_path)
        from kennel.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        add_task(tmp_path, title="threaded task", thread=thread)

        mock_github = MagicMock()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = []

        Cmd(github=mock_github).complete(tmp_path, "threaded task")

        mock_github.resolve_thread.assert_not_called()

    def test_completes_task_with_thread_exception_silenced(
        self, tmp_path: Path, caplog
    ) -> None:
        _task_file(tmp_path)
        from kennel.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        add_task(tmp_path, title="threaded task", thread=thread)

        mock_github = MagicMock()
        mock_github.get_user.side_effect = RuntimeError("network error")

        # Should not raise; exception is swallowed and logged
        with caplog.at_level(logging.WARNING, logger="kennel"):
            Cmd(github=mock_github).complete(tmp_path, "threaded task")
        assert "thread resolution skipped" in caplog.text

    def test_completes_task_with_thread_already_resolved(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        from kennel.tasks import add_task

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        add_task(tmp_path, title="threaded task", thread=thread)

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
        mock_github.get_review_threads.return_value = {
            "data": {
                "repository": {
                    "pullRequest": {
                        "reviewThreads": {
                            "nodes": [
                                {
                                    "id": "thread_node_abc",
                                    "isResolved": True,
                                    "comments": {"nodes": [{"databaseId": 42}]},
                                }
                            ]
                        }
                    }
                }
            }
        }

        Cmd(github=mock_github).complete(tmp_path, "threaded task")

        mock_github.resolve_thread.assert_not_called()

    def test_thread_missing_fields_skips(self, tmp_path: Path) -> None:
        """Thread dict with missing fields should silently skip resolution."""
        _task_file(tmp_path)
        from kennel.tasks import add_task

        # thread missing 'pr' and 'comment_id'
        add_task(tmp_path, title="task", thread={"repo": "a/b"})

        mock_github = MagicMock()
        Cmd(github=mock_github).complete(tmp_path, "task")

        mock_github.resolve_thread.assert_not_called()


# ── Cmd.list ──────────────────────────────────────────────────────────────────


class TestCmdList:
    def test_prints_json(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        cmd = Cmd()
        cmd.add(tmp_path, "alpha", "")
        cmd.add(tmp_path, "beta", "desc")
        cmd.list(tmp_path)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2
        assert data[0]["title"] == "alpha"
        assert data[1]["title"] == "beta"

    def test_empty_list(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        Cmd().list(tmp_path)
        out = capsys.readouterr().out
        assert json.loads(out) == []


# ── main (integration) ────────────────────────────────────────────────────────


class TestMain:
    def test_add_via_main(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "task title"])
        from kennel.tasks import list_tasks

        tasks = list_tasks(tmp_path)
        assert tasks[0]["title"] == "task title"

    def test_complete_via_main(self, tmp_path: Path) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "finish me"])
        main([str(tmp_path), "complete", "finish me"])
        from kennel.tasks import list_tasks

        assert list_tasks(tmp_path)[0]["status"] == "completed"

    def test_list_via_main(self, tmp_path: Path, capsys) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "one"])
        main([str(tmp_path), "list"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["title"] == "one"

    def test_no_args_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_raises(self, tmp_path: Path, monkeypatch) -> None:
        """Fallback case in match statement raises AssertionError."""
        from unittest.mock import MagicMock

        import kennel.cli as cli_mod

        fake_args = MagicMock()
        fake_args.command = "bogus"
        fake_parser = MagicMock()
        fake_parser.parse_args.return_value = fake_args

        monkeypatch.setattr(cli_mod, "build_parser", lambda: fake_parser)
        with pytest.raises(AssertionError, match="unreachable"):
            main([])
