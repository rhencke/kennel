import asyncio
from pathlib import Path

from io_boundaries import (
    IO,
    _io_http_status_ok,
    _io_read_file_echo,
    http_status_ok,
    read_file_echo,
)


def test_file_style_io_boundary_extracts_to_async_facade(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("hello from fido\n", encoding="utf-8")

    assert asyncio.run(read_file_echo(str(payload))) == "hello from fido\n"

    io_value = _io_read_file_echo(str(payload))
    assert isinstance(io_value, IO)
    assert asyncio.run(io_value.run()) == "hello from fido\n"


def test_http_style_io_boundary_maps_effect_result() -> None:
    assert asyncio.run(http_status_ok("https://ok.example")) is True
    assert asyncio.run(http_status_ok("https://not-ok.example")) is False

    io_value = _io_http_status_ok("https://ok.example")
    assert isinstance(io_value, IO)
    assert asyncio.run(io_value.run()) is True


def test_generated_io_source_uses_async_await(build_default: Path) -> None:
    source = (build_default / "io_boundaries.py").read_text()

    assert "def _io_read_file_echo" in source
    assert "async def read_file_echo" in source
    assert "return await _io_read_file_echo(path).run()" in source
    assert "IO.from_sync" in source
