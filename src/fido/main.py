"""Top-level fido server entry point."""

from collections.abc import Callable

from fido.server import run as _default_server_run


def main(
    argv: list[str] | None = None,
    *,
    _server_run: Callable[[], None] = _default_server_run,
) -> None:
    del argv
    _server_run()


if __name__ == "__main__":  # pragma: no cover
    main()
