"""Hand-rolled test fakes shared across the Fido test suite."""

from collections.abc import Callable


class _FakeDispatcher:
    """Hand-rolled fake for :class:`fido.events.Dispatcher`.

    Records calls with typed fields so tests can assert on sequencing,
    call counts, and arguments without MagicMock.  Configurable return
    values and side-effects cover the three usage patterns across the
    test suite:

    - ``dispatch_return`` / ``dispatch_side_effect`` for server tests
    - ``backfill_return`` / ``backfill_side_effect`` for worker tests
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
    ) -> None:
        self.dispatch_calls: list[tuple] = []
        self.backfill_calls: list[dict[str, object]] = []
        self.launch_sync_calls: int = 0
        self._dispatch_return = dispatch_return
        self._dispatch_side_effect = dispatch_side_effect
        self._backfill_return = backfill_return
        self._backfill_side_effect = backfill_side_effect

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

    def backfill_missed_pr_comments(self, pr_number: int, *, gh_user: str) -> int:
        self.backfill_calls.append({"pr_number": pr_number, "gh_user": gh_user})
        if callable(self._backfill_side_effect):
            return self._backfill_side_effect(pr_number, gh_user=gh_user)
        if isinstance(self._backfill_side_effect, BaseException):
            raise self._backfill_side_effect
        return self._backfill_return

    def launch_sync(self) -> None:
        self.launch_sync_calls += 1
