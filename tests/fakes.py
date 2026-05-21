"""Hand-rolled test fakes shared across the Fido test suite."""

from collections.abc import Callable
from pathlib import Path


class _FakeDispatcher:
    """Hand-rolled fake for :class:`fido.events.Dispatcher`.

    Records calls with typed fields so tests can assert on sequencing,
    call counts, and arguments without MagicMock.  Configurable return
    values and side-effects cover the four usage patterns across the
    test suite:

    - ``dispatch_return`` / ``dispatch_side_effect`` for server tests
    - ``backfill_return`` / ``backfill_side_effect`` for worker tests
    - ``reply_to_comment_return`` / ``reply_to_comment_side_effect`` for
      handler triage tests
    - ``reply_to_issue_comment_return`` / ``reply_to_issue_comment_side_effect``
      for issue-comment triage tests
    - bare construction for ``create_task`` tests that only need
      ``launch_sync()`` to be callable
    """

    def __init__(
        self,
        *,
        dispatch_return: object = None,
        dispatch_side_effect: Callable[..., object] | BaseException | None = None,
        backfill_return: int = 0,
        backfill_side_effect: (Callable[..., int] | BaseException | None) = None,
        recover_return: bool = False,
        recover_side_effect: (Callable[..., bool] | BaseException | None) = None,
        reply_to_comment_return: tuple[str, list[str]] = ("ANSWER", []),
        reply_to_comment_side_effect: (
            Callable[..., tuple[str, list[str]]] | BaseException | None
        ) = None,
        reply_to_issue_comment_return: tuple[str, list[str]] = ("ANSWER", []),
        reply_to_issue_comment_side_effect: (
            Callable[..., tuple[str, list[str]]] | BaseException | None
        ) = None,
    ) -> None:
        self.dispatch_calls: list[tuple] = []
        self.backfill_calls: list[dict[str, object]] = []
        self.launch_sync_calls: int = 0
        self.reorder_tasks_background_calls: list[tuple] = []
        self.recover_reply_promises_calls: list[dict[str, object]] = []
        self.reply_to_comment_calls: list[tuple] = []
        self.reply_to_issue_comment_calls: list[tuple] = []
        self._dispatch_return = dispatch_return
        self._dispatch_side_effect = dispatch_side_effect
        self._backfill_return = backfill_return
        self._backfill_side_effect = backfill_side_effect
        self._recover_return = recover_return
        self._recover_side_effect = recover_side_effect
        self._reply_to_comment_return = reply_to_comment_return
        self._reply_to_comment_side_effect = reply_to_comment_side_effect
        self._reply_to_issue_comment_return = reply_to_issue_comment_return
        self._reply_to_issue_comment_side_effect = reply_to_issue_comment_side_effect

    def dispatch(
        self,
        event: str,
        payload: dict,
        *,
        delivery_id: str | None = None,
        oracle: object = None,
    ) -> object:
        self.dispatch_calls.append((event, payload, delivery_id, oracle))
        if callable(self._dispatch_side_effect):
            return self._dispatch_side_effect(
                event, payload, delivery_id=delivery_id, oracle=oracle
            )
        if isinstance(self._dispatch_side_effect, BaseException):
            raise self._dispatch_side_effect
        return self._dispatch_return

    def backfill_missed_pr_comments(
        self,
        pr_number: int,
        *,
        gh_user: str,
        registry: object = None,
        agent: object = None,
        prompts: object = None,
    ) -> int:
        # ``registry``/``agent``/``prompts`` arrived with #1814 so the
        # backfill can route through the live synthesis path
        # (``reply_to_issue_comment``).  This fake records them so
        # tests asserting on call shape can verify the worker is
        # passing the right collaborators.
        del agent, prompts
        self.backfill_calls.append(
            {
                "pr_number": pr_number,
                "gh_user": gh_user,
                "registry": registry,
            }
        )
        if callable(self._backfill_side_effect):
            return self._backfill_side_effect(pr_number, gh_user=gh_user)
        if isinstance(self._backfill_side_effect, BaseException):
            raise self._backfill_side_effect
        return self._backfill_return

    def recover_reply_promises(
        self,
        fido_dir: Path,
        pr_number: int,
        registry: object,
        *,
        agent: object = None,
        prompts: object = None,
    ) -> bool:
        self.recover_reply_promises_calls.append(
            {
                "fido_dir": fido_dir,
                "pr_number": pr_number,
                "registry": registry,
                "agent": agent,
                "prompts": prompts,
            }
        )
        if callable(self._recover_side_effect):
            return self._recover_side_effect(
                fido_dir, pr_number, registry, agent=agent, prompts=prompts
            )
        if isinstance(self._recover_side_effect, BaseException):
            raise self._recover_side_effect
        return self._recover_return

    def reply_to_comment(
        self,
        action: object,
        registry: object,
    ) -> tuple[str, list[str]]:
        self.reply_to_comment_calls.append((action, registry))
        if callable(self._reply_to_comment_side_effect):
            return self._reply_to_comment_side_effect(action, registry)
        if isinstance(self._reply_to_comment_side_effect, BaseException):
            raise self._reply_to_comment_side_effect
        return self._reply_to_comment_return

    def reply_to_issue_comment(
        self,
        action: object,
        registry: object,
    ) -> tuple[str, list[str]]:
        self.reply_to_issue_comment_calls.append((action, registry))
        if callable(self._reply_to_issue_comment_side_effect):
            return self._reply_to_issue_comment_side_effect(action, registry)
        if isinstance(self._reply_to_issue_comment_side_effect, BaseException):
            raise self._reply_to_issue_comment_side_effect
        return self._reply_to_issue_comment_return

    def launch_sync(self) -> None:
        self.launch_sync_calls += 1

    def reorder_tasks_background(self, *args: object, **kwargs: object) -> None:
        self.reorder_tasks_background_calls.append((args, kwargs))


class _ReplyFakeDispatcher:
    """Proxy around a real :class:`fido.events.Dispatcher` that overrides only
    the reply methods for server integration tests.

    The ``dispatch()`` call (and all other Dispatcher methods not listed below)
    delegates to the wrapped real dispatcher so that Actions are created from
    the actual webhook payload.  Only ``reply_to_comment`` and
    ``reply_to_issue_comment`` are overridden — they return configurable values
    and record their calls without invoking the real synthesis path.

    Use this in tests that use the live ``server`` fixture (an actual HTTP
    server backed by a real ``Dispatcher`` for dispatch) but need to avoid
    hitting the real provider for the triage/reply step.
    """

    def __init__(
        self,
        real: object,
        *,
        reply_to_comment_return: tuple[str, list[str]] = ("ANSWER", []),
        reply_to_comment_side_effect: (
            Callable[..., tuple[str, list[str]]] | BaseException | None
        ) = None,
        reply_to_issue_comment_return: tuple[str, list[str]] = ("ANSWER", []),
        reply_to_issue_comment_side_effect: (
            Callable[..., tuple[str, list[str]]] | BaseException | None
        ) = None,
    ) -> None:
        self._real = real
        self.reply_to_comment_calls: list[tuple] = []
        self.reply_to_issue_comment_calls: list[tuple] = []
        self._reply_to_comment_return = reply_to_comment_return
        self._reply_to_comment_side_effect = reply_to_comment_side_effect
        self._reply_to_issue_comment_return = reply_to_issue_comment_return
        self._reply_to_issue_comment_side_effect = reply_to_issue_comment_side_effect

    def __getattr__(self, name: str) -> object:
        # Delegate everything not explicitly overridden to the real dispatcher
        # (dispatch, backfill_missed_pr_comments, recover_reply_promises, …).
        return getattr(self._real, name)

    def reply_to_comment(
        self,
        action: object,
        registry: object,
    ) -> tuple[str, list[str]]:
        self.reply_to_comment_calls.append((action, registry))
        if callable(self._reply_to_comment_side_effect):
            return self._reply_to_comment_side_effect(action, registry)
        if isinstance(self._reply_to_comment_side_effect, BaseException):
            raise self._reply_to_comment_side_effect
        return self._reply_to_comment_return

    def reply_to_issue_comment(
        self,
        action: object,
        registry: object,
    ) -> tuple[str, list[str]]:
        self.reply_to_issue_comment_calls.append((action, registry))
        if callable(self._reply_to_issue_comment_side_effect):
            return self._reply_to_issue_comment_side_effect(action, registry)
        if isinstance(self._reply_to_issue_comment_side_effect, BaseException):
            raise self._reply_to_issue_comment_side_effect
        return self._reply_to_issue_comment_return
