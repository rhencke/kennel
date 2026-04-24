import json
import os
import re
import subprocess
import textwrap
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


def test_structured_diagnostic_output_from_failed_extraction() -> None:
    if os.environ.get("FIDO_SKIP_DOCKER_HELPER_TESTS") == "1":
        pytest.skip("Docker helper is exercised outside the buildx pytest target")

    script = textwrap.dedent(
        """
        cat > test/diagnostics_negative_tmp.v <<'EOF'
        Declare ML Module "rocq-python-extraction".
        Declare ML Module "rocq-runtime.plugins.extraction".
        Extract Inductive nat => "int"
          [ "0" "(lambda x: x + 1)" ]
          "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".
        Definition bad_get (n : nat) : nat := n.
        Extract Constant bad_get => "__PYMONAD_STATE_GET__".
        Definition bad_monad_marker : nat := bad_get 0.
        Python Extraction bad_monad_marker.
        EOF
        sed -i "s/source_maps))/source_maps diagnostics_negative_tmp))/" test/dune
        opam exec -- dune build test/diagnostics_negative_tmp.vo
        """
    )

    result = subprocess.run(
        [
            str(REPO_ROOT / "rocq-python-extraction" / "run_in_docker.sh"),
            "rocq-python-extraction",
            "bash",
            "-euo",
            "pipefail",
            "-c",
            script,
        ],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    json_line = next(
        line.removeprefix("PYTHON_EXTRACTION_DIAGNOSTIC_JSON: ")
        for line in output.splitlines()
        if line.startswith("PYTHON_EXTRACTION_DIAGNOSTIC_JSON: ")
    )
    payload = json.loads(json_line)

    assert result.returncode != 0
    assert "Python ExtractionError [PYEX010]" in output
    assert "Remediation:" in output
    assert payload["version"] == 1
    assert payload["code"] == "PYEX010"
    assert payload["remediation"]
    assert payload["docs"].endswith("#pyex010-monad-marker-arity-mismatch")
