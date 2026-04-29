from pathlib import Path

from fido.rocq import replied_comment_claims as oracle


def _function_source(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    next_function = source.find("\ndef ", start + 1)
    if next_function == -1:
        return source[start:]
    return source[start:next_function]


def test_claim_all_list_traversal_lowers_to_for_loop() -> None:
    source = Path(oracle.__file__).read_text()
    claim_all = _function_source(source, "claim_all")

    assert "while True:" not in claim_all
    assert "for comment in comments:" in claim_all
    assert "return claim_all(" not in claim_all
