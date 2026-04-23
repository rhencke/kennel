import asyncio
from pathlib import Path

from future_done_after_set import Future as DoneFuture
from future_done_after_set import _io_future_done_after_set, future_done_after_set
from lock_channel_future_demo import (
    Channel,
    Future,
    Mutex,
    _io_lock_channel_future_demo,
    lock_channel_future_demo,
)

from fido.rocq.concurrency_primitives import _io_roundtrip_message, roundtrip_message


def test_lock_channel_future_demo_extracts_to_threading_wrappers() -> None:
    assert asyncio.run(lock_channel_future_demo()) == 7

    io_value = _io_lock_channel_future_demo
    assert asyncio.run(io_value.run()) == 7


def test_future_done_after_set_observes_completion() -> None:
    assert asyncio.run(future_done_after_set()) is True

    io_value = _io_future_done_after_set
    assert asyncio.run(io_value.run()) is True


def test_generated_concurrency_source_documents_runtime_assumptions(
    build_default: Path,
) -> None:
    source = (build_default / "lock_channel_future_demo.py").read_text()

    assert "import threading" in source
    assert "import queue" in source
    assert "from concurrent.futures import Future as _ConcurrentFuture" in source
    assert "class Mutex" in source
    assert "class Channel" in source
    assert "class Future" in source
    assert "does not prove fairness" in source
    assert "does not model scheduler fairness" in source
    assert "double completion raises" in source
    assert "from contextlib import asynccontextmanager" in source
    assert "async def ownership(" in source
    assert "async with cls.ownership(acquire, release) as owner:" in source
    assert "finally:" in source
    assert "IO.bracket(" in source
    assert "async def lock_channel_future_demo" in source
    assert "return await _io_lock_channel_future_demo.run()" in source


def test_wrapper_instances_are_explicit_runtime_types() -> None:
    mutex = asyncio.run(Mutex.new().run())
    channel = asyncio.run(Channel[int].new().run())
    future = asyncio.run(Future[int].new().run())

    assert isinstance(mutex, Mutex)
    assert isinstance(channel, Channel)
    assert isinstance(future, Future)
    assert not future.done()
    asyncio.run(future.set_result(3).run())
    assert future.done()
    assert asyncio.run(future.result().run()) == 3

    done_future = asyncio.run(DoneFuture[int].new().run())
    assert isinstance(done_future, DoneFuture)


def test_production_concurrency_model_extracts_and_runs() -> None:
    assert asyncio.run(roundtrip_message(11)) == 11
    assert asyncio.run(_io_roundtrip_message(12).run()) == 12
