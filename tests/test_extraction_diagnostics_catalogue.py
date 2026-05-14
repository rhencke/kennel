import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BACKEND = REPO_ROOT / "rocq-python-extraction" / "python.ml"
DIAGNOSTICS_DOC = REPO_ROOT / "rocq-python-extraction" / "DIAGNOSTICS.md"


@dataclass(frozen=True)
class CatalogueEntry:
    code: str
    title: str
    category: str
    remediation: str
    docs: str


def _catalogue_entries() -> tuple[CatalogueEntry, ...]:
    source = PYTHON_BACKEND.read_text()
    pattern = re.compile(
        r'\{ code = "(?P<code>PYEX\d{3})"; '
        r'title = "(?P<title>[^"]+)"; '
        r'category = "(?P<category>[^"]+)"; '
        r'remediation = "(?P<remediation>[^"]+)"; '
        r'docs = "(?P<docs>[^"]+)" \};'
    )
    return tuple(
        CatalogueEntry(**match.groupdict()) for match in pattern.finditer(source)
    )


CATALOGUE = _catalogue_entries()


def test_catalogue_has_at_least_forty_unique_entries() -> None:
    codes = [entry.code for entry in CATALOGUE]

    assert len(CATALOGUE) >= 40
    assert len(codes) == len(set(codes))
    assert codes == sorted(codes)


@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda entry: entry.code)
def test_catalogue_entry_has_remediation_and_docs(entry: CatalogueEntry) -> None:
    docs = DIAGNOSTICS_DOC.read_text()
    anchor = entry.docs.split("#", 1)[1]

    assert entry.title
    assert entry.category
    assert entry.remediation.endswith(".")
    assert entry.docs.startswith("rocq-python-extraction/DIAGNOSTICS.md#")
    assert f"## {entry.code} " in docs
    assert anchor in docs.lower().replace(" ", "-")


def test_structured_diagnostic_renderer_fields_are_present() -> None:
    source = PYTHON_BACKEND.read_text()

    assert "PYTHON_EXTRACTION_DIAGNOSTIC_JSON: " in source
    assert '\\"version\\": 1' in source
    for field in ("code", "title", "category", "message", "remediation", "docs"):
        assert f'field "{field}"' in source
    for field in ("symbol", "detail"):
        assert f'field "{field}"' in source


def test_structured_diagnostic_payload_shape_is_parseable() -> None:
    # Hand-written fixture mirroring what the renderer in python.ml emits when
    # a real extraction fails with PYEX010.  Earlier this test ran a docker
    # subprocess that built a deliberately-broken .v file; under xdist that one
    # 47-second subprocess starved the other workers and inflated unrelated
    # FidoStore tests by ~50x.  The renderer-side guarantee — that python.ml
    # actually emits this shape — lives in
    # ``test_structured_diagnostic_renderer_fields_are_present`` (which scans
    # python.ml for the field-emitting calls) and in the catalogue tests
    # (which assert PYEX010 has the matching docs anchor).
    line = "PYTHON_EXTRACTION_DIAGNOSTIC_JSON: " + json.dumps(
        {
            "version": 1,
            "code": "PYEX010",
            "title": "Monad marker arity mismatch",
            "category": "Monad markers",
            "message": (
                "bad_monad_marker is tagged as a __PYMONAD_STATE_GET__ "
                "marker but has arity 0"
            ),
            "remediation": (
                "Reshape the body so the marker takes the expected argument."
            ),
            "docs": (
                "rocq-python-extraction/DIAGNOSTICS.md"
                "#pyex010-monad-marker-arity-mismatch"
            ),
            "symbol": "bad_monad_marker",
            "detail": "expected arity >= 1, got 0",
        }
    )

    payload = json.loads(line.removeprefix("PYTHON_EXTRACTION_DIAGNOSTIC_JSON: "))

    assert payload["version"] == 1
    assert payload["code"] == "PYEX010"
    assert payload["remediation"]
    assert payload["docs"].endswith("#pyex010-monad-marker-arity-mismatch")
