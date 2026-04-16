from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kennel.provider import ProviderModel
from kennel.session_agent import SessionBackedAgent


class _FakeAgent(SessionBackedAgent):
    voice_model = ProviderModel("voice")
    brief_model = ProviderModel("brief")

    def __init__(
        self,
        *,
        session_fn=lambda: None,
        session_system_file: Path | None = None,
        work_dir: Path | str | None = None,
        repo_name: str | None = None,
        session=None,
        session_factory=None,
    ) -> None:
        self._session_factory = (
            MagicMock() if session_factory is None else session_factory
        )
        super().__init__(
            session_fn=session_fn,
            session_system_file=session_system_file,
            work_dir=work_dir,
            repo_name=repo_name,
            session=session,
        )

    def _spawn_owned_session(self, model: ProviderModel):
        return self._session_factory(model)

    def run_turn(
        self,
        content: str,
        *,
        model: ProviderModel | None = None,
        system_prompt: str | None = None,
        retry_on_preempt: bool = False,
        fresh_session: bool = False,
    ) -> str:
        del retry_on_preempt
        session = self._resolve_turn_session(model=model, fresh_session=fresh_session)
        return session.prompt(content, model=model, system_prompt=system_prompt)


class TestSessionBackedAgent:
    def test_base_abstract_methods_raise(self) -> None:
        agent = SessionBackedAgent(
            session_fn=lambda: None,
            session_system_file=None,
            work_dir=None,
            repo_name=None,
            session=None,
        )
        with pytest.raises(NotImplementedError):
            agent.run_turn("hi")
        with pytest.raises(NotImplementedError):
            agent._spawn_owned_session(ProviderModel("model"))

    def test_session_properties_attach_and_detach(self) -> None:
        session = MagicMock(owner="worker", pid=123, session_id="sess-1")
        session.is_alive.return_value = True
        agent = _FakeAgent(session=session)
        assert agent.session is session
        assert agent.session_owner == "worker"
        assert agent.session_alive is True
        assert agent.session_pid == 123
        assert agent.session_id == "sess-1"
        assert agent.detach_session() is session
        assert agent.session is None

    def test_session_id_none_branches(self) -> None:
        assert _FakeAgent().session_id is None
        assert _FakeAgent(session=object()).session_id is None
        assert (
            _FakeAgent(session=type("S", (), {"session_id": 123})()).session_id is None
        )

    def test_ensure_session_requires_factory_inputs(self) -> None:
        with pytest.raises(
            ValueError,
            match="_FakeAgent.ensure_session requires session_system_file and work_dir",
        ):
            _FakeAgent().ensure_session()

    def test_ensure_session_requires_model_when_creating_session(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ValueError,
            match="_FakeAgent.ensure_session requires model when creating a session",
        ):
            _FakeAgent(
                session_system_file=tmp_path / "persona.md",
                work_dir=tmp_path,
            ).ensure_session()

    def test_ensure_session_spawns_owned_and_switches_existing(
        self, tmp_path: Path
    ) -> None:
        spawned = MagicMock()
        factory = MagicMock(return_value=spawned)
        agent = _FakeAgent(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
            session_factory=factory,
        )
        agent.ensure_session(agent.voice_model)
        factory.assert_called_once_with(agent.voice_model)
        attached = MagicMock()
        agent = _FakeAgent(session=attached)
        agent.ensure_session(agent.brief_model)
        attached.switch_model.assert_called_once_with(agent.brief_model)

    def test_resolve_turn_prefers_live_session_over_owned_spawn(
        self, tmp_path: Path
    ) -> None:
        resolved = MagicMock()
        resolved.prompt.return_value = "ok"
        factory = MagicMock()
        agent = _FakeAgent(
            session_fn=lambda: resolved,
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
            session_factory=factory,
        )
        assert agent.generate_reply("hi") == "ok"
        factory.assert_not_called()

    def test_run_turn_requires_model_when_spawning_owned_session(
        self, tmp_path: Path
    ) -> None:
        agent = _FakeAgent(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        with pytest.raises(
            ValueError,
            match="_FakeAgent.run_turn requires model when creating a session",
        ):
            agent.run_turn("hi")

    def test_resolve_turn_raises_resolver_error_or_generic_error(self) -> None:
        agent = _FakeAgent(
            session_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError, match="boom"):
            agent.generate_reply("hi")
        with pytest.raises(
            RuntimeError, match="_FakeAgent.run_turn could not resolve a session"
        ):
            _FakeAgent(session_fn=lambda: None).generate_reply("hi")

    def test_fresh_session_requires_resettable_session(self) -> None:
        agent = _FakeAgent(session=object())
        with pytest.raises(
            ValueError,
            match="_FakeAgent.run_turn fresh_session requires resettable session",
        ):
            agent.run_turn("hi", model=agent.voice_model, fresh_session=True)

    def test_shared_helper_methods_and_json_parsing(self) -> None:
        session = MagicMock(session_id="sess-1")
        session.prompt.side_effect = [
            "reply text",
            "branch-name\nextra",
            "status text",
            '{"emoji":"rocket"}',
            "status with session",
            "resumed status",
        ]
        agent = _FakeAgent(session=session)
        assert agent.generate_reply("reply") == "reply text"
        assert agent.generate_branch_name("branch") == "branch-name"
        assert agent.generate_status("status", "system") == "status text"
        assert agent.generate_status_emoji("emoji", "system") == "rocket"
        assert agent.generate_status_with_session("status", "system") == (
            "status with session",
            "sess-1",
        )
        assert agent.resume_status("sess-1", "resume") == "resumed status"

    def test_status_emoji_returns_empty_on_empty_or_bad_json(self) -> None:
        session = MagicMock()
        session.prompt.side_effect = ["", "not json"]
        agent = _FakeAgent(session=session)
        assert agent.generate_status_emoji("emoji", "system") == ""
        assert agent.generate_status_emoji("emoji", "system") == ""

    def test_resume_status_requires_matching_live_session(self) -> None:
        session = MagicMock(session_id="other")
        agent = _FakeAgent(session=session)
        with pytest.raises(
            RuntimeError,
            match="_FakeAgent resume helpers require the matching live session",
        ):
            agent.resume_status("sess-1", "resume")

    def test_resume_status_uses_resolved_session_when_not_attached(self) -> None:
        session = MagicMock(session_id="sess-1")
        session.prompt.return_value = "resumed status"
        agent = _FakeAgent(session_fn=lambda: session)
        assert agent.resume_status("sess-1", "resume") == "resumed status"
