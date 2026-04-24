from http_status_ok import IO as HttpIO
from http_status_ok import _io_http_status_ok, http_status_ok
from read_file_echo import IO as ReadIO
from read_file_echo import _io_read_file_echo, read_file_echo


async def check_file_boundary(path: str) -> None:
    text: str = await read_file_echo(path)
    io_text: ReadIO[str] = _io_read_file_echo(path)
    text_again: str = await io_text.run()

    assert isinstance(text, str)
    assert isinstance(text_again, str)


async def check_http_boundary() -> None:
    ok: bool = await http_status_ok("https://ok.example")
    io_ok: HttpIO[bool] = _io_http_status_ok("https://ok.example")
    ok_again: bool = await io_ok.run()

    assert isinstance(ok, bool)
    assert isinstance(ok_again, bool)
