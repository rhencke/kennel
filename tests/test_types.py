"""Tests for shared type definitions in :mod:`fido.types`.

Currently focused on :class:`IntentVerdict` (#1798 / INV-D) — the
per-intent verdict shape Opus emits as part of a rescope batch.  The
verdict drives the INV-E (#1803) reply-back classifier downstream.
"""

import dataclasses

from fido.types import IntentVerdict


class TestIntentVerdictShape:
    def test_minimal_construction_with_outcome_honored(self) -> None:
        """A fully-honored intent: just the intent id + outcome."""
        v = IntentVerdict(intent_comment_id=42, outcome="honored")
        assert v.intent_comment_id == 42
        assert v.outcome == "honored"
        assert v.ops == []
        assert v.affected_task_ids == []
        assert v.by_intent_comment_id is None
        assert v.narrative is None

    def test_construction_with_all_fields(self) -> None:
        v = IntentVerdict(
            intent_comment_id=100,
            outcome="reshaped",
            ops=[{"op": "rewrite", "id": "T1", "title": "new"}],
            affected_task_ids=["T1"],
            by_intent_comment_id=None,
            narrative="Folded the ask into the existing parser task.",
        )
        assert v.outcome == "reshaped"
        assert v.ops == [{"op": "rewrite", "id": "T1", "title": "new"}]
        assert v.affected_task_ids == ["T1"]
        assert v.narrative is not None

    def test_supersedence_pointer_within_batch(self) -> None:
        # "red" → "no, green" — red's verdict points at green's intent
        # comment id.  by_intent_comment_id is set; ops may be empty
        # (full supersedence) or partial (component supersedence).
        v = IntentVerdict(
            intent_comment_id=10,
            outcome="superseded",
            by_intent_comment_id=11,
            narrative="Color request was overridden by a later comment.",
        )
        assert v.by_intent_comment_id == 11
        assert v.outcome == "superseded"

    def test_partial_supersedence_keeps_ops(self) -> None:
        # "paint and make it red" → "no, green": the paint op survives
        # (still honored), the color component is superseded.  Ops AND
        # by_intent_comment_id coexist.
        v = IntentVerdict(
            intent_comment_id=10,
            outcome="superseded",
            ops=[{"op": "new", "title": "Paint the surface", "type": "spec"}],
            affected_task_ids=["T-paint"],
            by_intent_comment_id=11,
            narrative="Paint kept; color component superseded by later comment.",
        )
        assert v.ops != []
        assert v.by_intent_comment_id == 11

    def test_jointly_honored_intents_share_task_id(self) -> None:
        # Canonical 3+1 reviewer pattern: three comments asking the
        # same fix + a fourth "just fix all of these" all attribute
        # to ONE consolidated task.  Each verdict can list the same
        # task id in affected_task_ids.
        verdicts = [
            IntentVerdict(
                intent_comment_id=cid,
                outcome="honored",
                ops=[],
                affected_task_ids=["T-fix-all"],
            )
            for cid in (1, 2, 3, 4)
        ]
        for v in verdicts:
            assert "T-fix-all" in v.affected_task_ids

    def test_no_op_outcome(self) -> None:
        # Acknowledgement / "request already covered" verdict.  No ops,
        # no affected tasks, no narrative needed.
        v = IntentVerdict(intent_comment_id=999, outcome="no_op")
        assert v.outcome == "no_op"
        assert v.ops == []
        assert v.affected_task_ids == []

    def test_frozen(self) -> None:
        # Verdict shape is frozen — once Opus emits it, the runtime
        # must not mutate.  Catches accidental in-place edits.
        v = IntentVerdict(intent_comment_id=1, outcome="honored")
        try:
            v.outcome = "reshaped"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("IntentVerdict must be frozen")

    def test_default_factories_independent_per_instance(self) -> None:
        # ``field(default_factory=list)`` guard: two default-constructed
        # verdicts must not share the same ``ops`` / ``affected_task_ids``
        # list object, or a mutation on one would corrupt the other.
        a = IntentVerdict(intent_comment_id=1, outcome="honored")
        b = IntentVerdict(intent_comment_id=2, outcome="honored")
        assert a.ops is not b.ops
        assert a.affected_task_ids is not b.affected_task_ids
