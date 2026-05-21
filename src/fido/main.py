"""Top-level fido server entry point."""


def main() -> None:  # pragma: no cover
    from fido.server import run

    run()


if __name__ == "__main__":  # pragma: no cover
    main()
