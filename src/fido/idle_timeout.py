"""Shared idle-timeout bookkeeping for streaming provider turns."""

import time
from collections.abc import Callable


class IdleDeadline:
    """Track a timeout that resets whenever provider activity arrives."""

    def __init__(
        self,
        timeout: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout = timeout
        self._clock = clock
        self._last_activity = self._clock()

    def reset(self) -> None:
        self._last_activity = self._clock()

    def remaining(self) -> float:
        return self._timeout - (self._clock() - self._last_activity)
