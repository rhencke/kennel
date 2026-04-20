import pytest
from proof_pair_zero import _Impossible, proof_pair_zero


def test_proof_pair_zero_uses_impossible_witness(build_default) -> None:
    with pytest.raises(_Impossible):
        proof_pair_zero(0)

    source = (build_default / "proof_pair_zero.py").read_text()
    assert "_impossible()" in source
