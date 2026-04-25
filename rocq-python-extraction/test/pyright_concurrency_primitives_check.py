from typing import cast

from concurrency_primitives import (
    IO,
    Channel,
    Future,
    Mutex,
    _io_future_done_after_set,
    _io_lock_channel_future_demo,
    future_done_after_set,
    lock_channel_future_demo,
)


async def check_lock_channel_future_demo() -> None:
    result: int = await lock_channel_future_demo()
    io_result: IO[int] = _io_lock_channel_future_demo

    mutex_io: IO[Mutex] = Mutex.new()
    channel_io: IO[Channel[int]] = Channel[int].new()
    future_io: IO[Future[int]] = Future[int].new()

    mutex = cast(Mutex, object())
    channel = cast(Channel[int], object())
    future = cast(Future[int], object())

    sent: IO[None] = channel.send(result)
    received: IO[int] = channel.receive()

    acquired: IO[None] = mutex.acquire()
    released: IO[None] = mutex.release()
    completed_set: IO[None] = future.set_result(result)
    done: bool = future.done()
    completed: IO[int] = future.result()
    bracketed: IO[int] = IO.bracket(
        mutex.acquire(),
        lambda _owner: mutex.release(),
        lambda _owner: completed,
    )

    assert isinstance(io_result, IO)
    assert isinstance(mutex_io, IO)
    assert isinstance(channel_io, IO)
    assert isinstance(future_io, IO)
    assert isinstance(sent, IO)
    assert isinstance(received, IO)
    assert isinstance(acquired, IO)
    assert isinstance(released, IO)
    assert isinstance(completed_set, IO)
    assert isinstance(done, bool)
    assert isinstance(completed, IO)
    assert isinstance(bracketed, IO)


async def check_future_done_after_set() -> None:
    done: bool = await future_done_after_set()
    io_done: IO[bool] = _io_future_done_after_set
    future_io: IO[Future[int]] = Future[int].new()

    assert isinstance(done, bool)
    assert isinstance(io_done, IO)
    assert isinstance(future_io, IO)
