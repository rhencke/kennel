from collections import Counter
from pathlib import Path

GENERATED_PYTEST_TARGETS = Path(
    "rocq-python-extraction/test/generated_pytest_targets.txt"
)


def test_generated_typevar_declarations_are_unique(build_default: Path) -> None:
    for target in GENERATED_PYTEST_TARGETS.read_text().splitlines():
        source = (build_default / target).read_text()
        declarations = [
            line.strip() for line in source.splitlines() if " = TypeVar(" in line
        ]
        duplicates = [
            declaration
            for declaration, count in Counter(declarations).items()
            if count > 1
        ]

        assert duplicates == [], (
            f"{target} has duplicate TypeVar declarations: {duplicates}"
        )
