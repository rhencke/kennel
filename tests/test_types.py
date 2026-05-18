"""Tests for shared type definitions in :mod:`fido.types`.

Currently focused on :class:`IntentVerdict` (#1798 / INV-D) — the
per-intent verdict shape Opus emits as part of a rescope batch.  The
verdict drives the INV-E (#1803) reply-back classifier downstream.
"""

import dataclasses

import pytest

from fido.types import IntentVerdict


class TestIntentVerdictShape:
    def test_minimal_construction_with_outcome_honored(self) -> None:
        """A fully-honored intent: just the intent id + outcome."""
        v = IntentVerdict(intent_comment_id=42, outcome="honored")
        assert v.intent_comment_id == 42
        assert v.outcome == "honored"
        assert v.ops == ()
        assert v.affected_task_ids == ()
        assert v.by_intent_comment_id is None
        assert v.narrative is None

    def test_construction_with_all_fields(self) -> None:
        v = IntentVerdict(
            intent_comment_id=100,
            outcome="reshaped",
            ops=({"op": "rewrite", "id": "T1", "title": "new"},),
            affected_task_ids=("T1",),
            by_intent_comment_id=None,
            narrative="Folded the ask into the existing parser task.",
        )
        assert v.outcome == "reshaped"
        assert v.ops == ({"op": "rewrite", "id": "T1", "title": "new"},)
        assert v.affected_task_ids == ("T1",)
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
            ops=({"op": "new", "title": "Paint the surface", "type": "spec"},),
            affected_task_ids=("T-paint",),
            by_intent_comment_id=11,
            narrative="Paint kept; color component superseded by later comment.",
        )
        assert v.ops != ()
        assert v.by_intent_comment_id == 11

    def test_jointly_honored_intents_share_task_id(self) -> None:
        # Canonical 3+1 reviewer pattern: three comments asking the
        # same fix + a fourth "just fix all of these" all attribute
        # to ONE consolidated task.  Each verdict can list the same
        # task id in affected_task_ids.
        verdicts = tuple(
            IntentVerdict(
                intent_comment_id=cid,
                outcome="honored",
                ops=(),
                affected_task_ids=("T-fix-all",),
            )
            for cid in (1, 2, 3, 4)
        )
        for v in verdicts:
            assert "T-fix-all" in v.affected_task_ids

    def test_no_op_outcome(self) -> None:
        # Acknowledgement / "request already covered" verdict.  No ops,
        # no affected tasks, no narrative needed.
        v = IntentVerdict(intent_comment_id=999, outcome="no_op")
        assert v.outcome == "no_op"
        assert v.ops == ()
        assert v.affected_task_ids == ()

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

    def test_ops_is_tuple_not_list(self) -> None:
        # codex P1 on #1802: deep immutability — ops as a list lets
        # callers ``verdict.ops.append(...)`` past the frozen guard.
        # Tuple makes the in-place mutation a TypeError at boundary.
        v = IntentVerdict(intent_comment_id=1, outcome="honored")
        assert isinstance(v.ops, tuple)
        assert isinstance(v.affected_task_ids, tuple)

    def test_ops_list_input_coerced_to_tuple(self) -> None:
        # codex P1 on #1802 (round 2): the tuple annotation isn't
        # enforced at runtime.  Callers can pass a plain list; the
        # post-init coerces.  Mutating the original list after
        # construction MUST NOT bleed into the verdict.
        original_ops = [{"op": "rewrite", "id": "T1"}]
        v = IntentVerdict(
            intent_comment_id=1,
            outcome="reshaped",
            ops=original_ops,  # type: ignore[arg-type]
            affected_task_ids=["T1"],  # type: ignore[arg-type]
            narrative="x",
        )
        original_ops.append({"op": "remove", "id": "T2"})
        assert isinstance(v.ops, tuple)
        assert len(v.ops) == 1
        assert isinstance(v.affected_task_ids, tuple)

    def test_op_payload_dicts_are_frozen(self) -> None:
        # codex P2 on #1802: even with tuple ops, each ``dict`` was
        # still mutable.  Deep-freeze coerces to frozendict so
        # ``v.ops[0]["op"] = "x"`` raises at the boundary.
        v = IntentVerdict(
            intent_comment_id=1,
            outcome="reshaped",
            ops=({"op": "rewrite", "id": "T1"},),
            affected_task_ids=("T1",),
            narrative="x",
        )
        from collections.abc import Mapping

        assert isinstance(v.ops[0], Mapping)
        # frozendict raises TypeError on item assignment.
        with pytest.raises((TypeError, AttributeError)):
            v.ops[0]["op"] = "remove"  # type: ignore[index]

    def test_default_collections_are_singleton_empty_tuples(self) -> None:
        # Tuples are immutable so two verdicts sharing the default
        # empty tuple is safe (no foot-gun like a shared default
        # ``[]`` would have been).
        a = IntentVerdict(intent_comment_id=1, outcome="honored")
        b = IntentVerdict(intent_comment_id=2, outcome="honored")
        assert a.ops == b.ops == ()
        assert a.affected_task_ids == b.affected_task_ids == ()


class TestIntentVerdictValidation:
    def test_self_supersedence_rejected(self) -> None:
        # codex P2 on #1802: by_intent_comment_id pointing at the
        # verdict's own intent is meaningless and indicates a
        # malformed batch.  Fail at construction.
        with pytest.raises(ValueError, match="must not reference"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=10,
                narrative="self",
            )

    def test_superseded_outcome_requires_by_intent_comment_id(self) -> None:
        # codex P2 on #1802: superseded outcome without a pointer is
        # nonsense — INV-E can't tell self-vs-cross-author without it.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                narrative="something",
            )

    def test_reshaped_outcome_requires_narrative(self) -> None:
        # codex P2 on #1802: reshaped means material change → reply-back
        # fires → reply needs prose.  Empty narrative is a contract
        # violation at the boundary.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="reshaped",
                ops=({"op": "rewrite", "id": "T1", "title": "x"},),
                affected_task_ids=("T1",),
            )

    def test_superseded_outcome_requires_narrative(self) -> None:
        # Same reasoning as the reshaped case.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=11,
                narrative="",
            )

    def test_superseded_whitespace_only_narrative_rejected(self) -> None:
        # Narrative must be substantively non-empty.  "   " is empty
        # for human-facing reply purposes.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=11,
                narrative="   ",
            )

    def test_honored_outcome_allows_blank_narrative(self) -> None:
        # honored never warrants reply-back; narrative is optional.
        v = IntentVerdict(intent_comment_id=10, outcome="honored")
        assert v.narrative is None

    def test_no_op_outcome_allows_blank_narrative(self) -> None:
        # no_op same as honored — never warrants reply-back.
        v = IntentVerdict(intent_comment_id=10, outcome="no_op")
        assert v.narrative is None

    def test_by_intent_comment_id_rejected_on_honored(self) -> None:
        # codex P2 round 2 on #1802: only ``superseded`` carries a
        # supersedence pointer.  ``honored`` + ``by_intent_comment_id``
        # is contradictory metadata; reject at boundary.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                by_intent_comment_id=2,
            )

    def test_by_intent_comment_id_rejected_on_reshaped(self) -> None:
        # ``reshaped`` is Opus's reframing of a single intent — no
        # other intent is involved, so the pointer is contradictory.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="reshaped",
                by_intent_comment_id=2,
                narrative="x",
            )

    def test_by_intent_comment_id_rejected_on_no_op(self) -> None:
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="no_op",
                by_intent_comment_id=2,
            )
