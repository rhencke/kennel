import asyncio
from pathlib import Path

from concurrency_primitives import (
    Channel,
    Future,
    Mutex,
    _io_future_done_after_set,
    _io_lock_channel_future_demo,
    future_done_after_set,
    lock_channel_future_demo,
)

from fido.rocq.concurrency_primitives import _io_roundtrip_message, roundtrip_message

DoneFuture = Future


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
    """Concurrency wrappers (Mutex, Channel, Future, ownership context
    manager) live in ``fido.rocq_runtime`` now and the extracted
    ``concurrency_primitives.py`` re-exports them via ``import *``.
    Assert both sides:

    1. The generated file imports from the runtime module and uses the
       runtime types (so the marker calls actually lower to those types).
    2. The runtime file documents the assumptions (no fairness proof,
       double completion behavior, etc.) that this test originally
       asserted on the generated file before the runtime moved.
    """
    generated = (build_default / "concurrency_primitives.py").read_text()
    assert "from fido.rocq_runtime import *" in generated
    assert "async def lock_channel_future_demo" in generated
    assert "return await _io_lock_channel_future_demo.run()" in generated
    assert "IO.bracket(" in generated

    runtime_path = (
        Path(__file__).resolve().parents[2] / "src" / "fido" / "rocq_runtime.py"
    )
    runtime = runtime_path.read_text()
    assert "import threading" in runtime
    assert "import queue" in runtime
    assert "from concurrent.futures import Future as _ConcurrentFuture" in runtime
    assert "class Mutex" in runtime
    assert "class Channel" in runtime
    assert "class Future" in runtime
    assert "does not prove fairness" in runtime
    assert "does not model scheduler fairness" in runtime
    assert "double completion raises" in runtime
    assert "from contextlib import asynccontextmanager" in runtime
    assert "async def ownership(" in runtime
    assert "async with cls.ownership(acquire, release) as owner:" in runtime
    assert "finally:" in runtime


def test_concurrency_marker_calls_lower_to_runtime_methods(build_default: Path) -> None:
    source = (build_default / "concurrency_primitives.py").read_text()

    for snippet in (
        "Mutex.new()",
        "Channel.new()",
        "Future.new()",
        "mutex.acquire()",
        "mutex.release()",
        "channel.send(7)",
        "channel.receive()",
        "future.set_result(value)",
        "future.result()",
        "future.done()",
    ):
        assert snippet in source
    assert "__PYCONC_" not in source


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
