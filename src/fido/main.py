"""Top-level fido server entry point."""


def main(argv: list[str] | None = None) -> None:
    del argv
    from fido.server import run as server_run

    server_run()


if __name__ == "__main__":  # pragma: no cover
    main()
