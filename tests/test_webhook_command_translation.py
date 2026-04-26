"""Tests for the Rocq-extracted webhook_command_translation module.

Round-trip tests verify that every extracted function behaves exactly as
the Rocq definitions specify.  Each test is named after the theorem (or
definition) it exercises so that a failure immediately identifies the
violated invariant.

Proved theorems covered:
  ``translate_total``              — every WebhookEvent yields a WebhookCommand
  ``translate_preserves_delivery`` — cmd_delivery_id(translate(ev)) == evt_delivery_id(ev)
  ``same_delivery_refl``           — same_delivery(cmd, cmd) is True
  ``same_delivery_sym``            — same_delivery(c1, c2) == same_delivery(c2, c1)
  ``cmd_to_contender_is_handler``  — every webhook command maps to Handler
"""

from fido.rocq.webhook_command_translation import (
    CIFailure,
    CITimedOut,
    CmdCIFailure,
    CmdComment,
    CmdIssueAssigned,
    CmdPRMerged,
    CmdReviewSubmitted,
    EvtCIFailure,
    EvtIssueAssigned,
    EvtIssueComment,
    EvtPRMerged,
    EvtReviewComment,
    EvtReviewSubmitted,
    Handler,
    ReviewLine,
    TopLevelPR,
    WebhookEvent,
    cmd_delivery_id,
    cmd_to_contender,
    evt_delivery_id,
    same_delivery,
    translate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D1 = 1  # synthetic delivery id for event 1
_D2 = 42  # synthetic delivery id for event 2


def _review_comment(d: int = _D1) -> EvtReviewComment:
    return EvtReviewComment(d, 7, 101, "owner", False)


def _issue_comment(d: int = _D1) -> EvtIssueComment:
    return EvtIssueComment(d, 7, 102, "owner", False)


def _ci_failure(d: int = _D1) -> EvtCIFailure:
    return EvtCIFailure(d, "test-suite", CIFailure(), [3, 5])


def _ci_timed_out(d: int = _D1) -> EvtCIFailure:
    return EvtCIFailure(d, "slow-check", CITimedOut(), [])


def _pr_merged(d: int = _D1) -> EvtPRMerged:
    return EvtPRMerged(d, 9)


def _issue_assigned(d: int = _D1) -> EvtIssueAssigned:
    return EvtIssueAssigned(d, 12, "fido")


def _review_submitted(d: int = _D1) -> EvtReviewSubmitted:
    return EvtReviewSubmitted(d, 7, 55, "owner")


# ---------------------------------------------------------------------------
# translate_total: every WebhookEvent maps to a WebhookCommand
# ---------------------------------------------------------------------------


class TestTranslateTotal:
    def test_review_comment_becomes_cmd_comment_review_line(self) -> None:
        """EvtReviewComment → CmdComment(ReviewLine).

        translate_total + structural equality for the ReviewLine kind.
        """
        cmd = translate(_review_comment())
        assert isinstance(cmd, CmdComment), "translate_total"
        assert isinstance(cmd.cmd_kind, ReviewLine), "translate_total"

    def test_issue_comment_becomes_cmd_comment_top_level_pr(self) -> None:
        """EvtIssueComment → CmdComment(TopLevelPR).

        translate_total + structural equality for the TopLevelPR kind.
        """
        cmd = translate(_issue_comment())
        assert isinstance(cmd, CmdComment), "translate_total"
        assert isinstance(cmd.cmd_kind, TopLevelPR), "translate_total"

    def test_ci_failure_becomes_cmd_ci_failure(self) -> None:
        """EvtCIFailure(CIFailure) → CmdCIFailure."""
        cmd = translate(_ci_failure())
        assert isinstance(cmd, CmdCIFailure), "translate_total"

    def test_ci_timed_out_becomes_cmd_ci_failure(self) -> None:
        """EvtCIFailure(CITimedOut) → CmdCIFailure."""
        cmd = translate(_ci_timed_out())
        assert isinstance(cmd, CmdCIFailure), "translate_total"

    def test_pr_merged_becomes_cmd_pr_merged(self) -> None:
        """EvtPRMerged → CmdPRMerged."""
        cmd = translate(_pr_merged())
        assert isinstance(cmd, CmdPRMerged), "translate_total"

    def test_issue_assigned_becomes_cmd_issue_assigned(self) -> None:
        """EvtIssueAssigned → CmdIssueAssigned."""
        cmd = translate(_issue_assigned())
        assert isinstance(cmd, CmdIssueAssigned), "translate_total"

    def test_review_submitted_becomes_cmd_review_submitted(self) -> None:
        """EvtReviewSubmitted → CmdReviewSubmitted."""
        cmd = translate(_review_submitted())
        assert isinstance(cmd, CmdReviewSubmitted), "translate_total"


# ---------------------------------------------------------------------------
# translate field threading
# ---------------------------------------------------------------------------


class TestTranslateFields:
    def test_review_comment_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtReviewComment."""
        cmd = translate(EvtReviewComment(99, 7, 101, "alice", True))
        assert isinstance(cmd, CmdComment)
        assert cmd.cmd_delivery == 99
        assert cmd.cmd_pr == 7
        assert cmd.cmd_comment_id == 101
        assert cmd.cmd_author == "alice"
        assert cmd.cmd_is_bot is True
        assert isinstance(cmd.cmd_kind, ReviewLine)

    def test_issue_comment_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtIssueComment."""
        cmd = translate(EvtIssueComment(88, 3, 202, "bob", False))
        assert isinstance(cmd, CmdComment)
        assert cmd.cmd_delivery == 88
        assert cmd.cmd_pr == 3
        assert cmd.cmd_comment_id == 202
        assert cmd.cmd_author == "bob"
        assert cmd.cmd_is_bot is False
        assert isinstance(cmd.cmd_kind, TopLevelPR)

    def test_ci_failure_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtCIFailure."""
        cmd = translate(EvtCIFailure(77, "lint", CIFailure(), [1, 2]))
        assert isinstance(cmd, CmdCIFailure)
        assert cmd.cmd_delivery == 77
        assert cmd.cmd_check_name == "lint"
        assert isinstance(cmd.cmd_conclusion, CIFailure)
        assert cmd.cmd_pr_numbers == [1, 2]

    def test_pr_merged_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtPRMerged."""
        cmd = translate(EvtPRMerged(66, 5))
        assert isinstance(cmd, CmdPRMerged)
        assert cmd.cmd_delivery == 66
        assert cmd.cmd_pr == 5

    def test_issue_assigned_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtIssueAssigned."""
        cmd = translate(EvtIssueAssigned(55, 12, "carol"))
        assert isinstance(cmd, CmdIssueAssigned)
        assert cmd.cmd_delivery == 55
        assert cmd.cmd_issue == 12
        assert cmd.cmd_assignee == "carol"

    def test_review_submitted_fields_preserved(self) -> None:
        """translate threads all payload fields from EvtReviewSubmitted."""
        cmd = translate(EvtReviewSubmitted(44, 7, 55, "dave"))
        assert isinstance(cmd, CmdReviewSubmitted)
        assert cmd.cmd_delivery == 44
        assert cmd.cmd_pr == 7
        assert cmd.cmd_review_id == 55
        assert cmd.cmd_author == "dave"


# ---------------------------------------------------------------------------
# translate_preserves_delivery
# ---------------------------------------------------------------------------


class TestTranslatePreservesDelivery:
    """cmd_delivery_id(translate(ev)) == evt_delivery_id(ev) for all ev."""

    def _assert_preserves(self, ev: WebhookEvent) -> None:
        cmd = translate(ev)
        assert cmd_delivery_id(cmd) == evt_delivery_id(ev), (
            "translate_preserves_delivery"
        )

    def test_review_comment(self) -> None:
        self._assert_preserves(EvtReviewComment(10, 1, 1, "u", False))

    def test_issue_comment(self) -> None:
        self._assert_preserves(EvtIssueComment(20, 1, 1, "u", False))

    def test_ci_failure(self) -> None:
        self._assert_preserves(EvtCIFailure(30, "c", CIFailure(), []))

    def test_pr_merged(self) -> None:
        self._assert_preserves(EvtPRMerged(40, 1))

    def test_issue_assigned(self) -> None:
        self._assert_preserves(EvtIssueAssigned(50, 1, "u"))

    def test_review_submitted(self) -> None:
        self._assert_preserves(EvtReviewSubmitted(60, 1, 1, "u"))


# ---------------------------------------------------------------------------
# same_delivery_refl: same_delivery(cmd, cmd) == True
# ---------------------------------------------------------------------------


class TestSameDeliveryRefl:
    """Every command is a duplicate of itself."""

    def _any_cmd(self) -> CmdPRMerged:
        return CmdPRMerged(1, 1)

    def test_same_delivery_refl(self) -> None:
        cmd = self._any_cmd()
        assert same_delivery(cmd, cmd) is True, "same_delivery_refl"

    def test_same_delivery_same_id_different_objects(self) -> None:
        """Two distinct command objects with equal delivery ids compare True."""
        c1 = CmdPRMerged(42, 1)
        c2 = CmdIssueAssigned(42, 1, "u")
        assert same_delivery(c1, c2) is True, "same_delivery_refl"

    def test_same_delivery_different_ids(self) -> None:
        """Commands with different delivery ids compare False."""
        c1 = CmdPRMerged(1, 1)
        c2 = CmdPRMerged(2, 1)
        assert same_delivery(c1, c2) is False


# ---------------------------------------------------------------------------
# same_delivery_sym: same_delivery(c1, c2) == same_delivery(c2, c1)
# ---------------------------------------------------------------------------


class TestSameDeliverySym:
    def test_same_id_is_symmetric(self) -> None:
        c1 = CmdPRMerged(7, 1)
        c2 = CmdIssueAssigned(7, 1, "u")
        assert same_delivery(c1, c2) == same_delivery(c2, c1), "same_delivery_sym"

    def test_different_id_is_symmetric(self) -> None:
        c1 = CmdPRMerged(1, 1)
        c2 = CmdPRMerged(9, 1)
        assert same_delivery(c1, c2) == same_delivery(c2, c1), "same_delivery_sym"


# ---------------------------------------------------------------------------
# cmd_to_contender_is_handler: every webhook command maps to Handler
# ---------------------------------------------------------------------------


class TestCmdToContenderIsHandler:
    """cmd_to_contender(cmd) == Handler for every WebhookCommand constructor."""

    def test_cmd_comment(self) -> None:
        cmd = CmdComment(1, 1, 1, "u", False, ReviewLine())
        assert isinstance(cmd_to_contender(cmd), Handler), "cmd_to_contender_is_handler"

    def test_cmd_ci_failure(self) -> None:
        cmd = CmdCIFailure(1, "c", CIFailure(), [])
        assert isinstance(cmd_to_contender(cmd), Handler), "cmd_to_contender_is_handler"

    def test_cmd_pr_merged(self) -> None:
        cmd = CmdPRMerged(1, 1)
        assert isinstance(cmd_to_contender(cmd), Handler), "cmd_to_contender_is_handler"

    def test_cmd_issue_assigned(self) -> None:
        cmd = CmdIssueAssigned(1, 1, "u")
        assert isinstance(cmd_to_contender(cmd), Handler), "cmd_to_contender_is_handler"

    def test_cmd_review_submitted(self) -> None:
        cmd = CmdReviewSubmitted(1, 1, 1, "u")
        assert isinstance(cmd_to_contender(cmd), Handler), "cmd_to_contender_is_handler"


# ---------------------------------------------------------------------------
# cmd_delivery_id: accessor extracts the right id from each constructor
# ---------------------------------------------------------------------------


class TestCmdDeliveryId:
    def test_cmd_comment(self) -> None:
        assert cmd_delivery_id(CmdComment(5, 1, 1, "u", False, TopLevelPR())) == 5

    def test_cmd_ci_failure(self) -> None:
        assert cmd_delivery_id(CmdCIFailure(6, "c", CITimedOut(), [])) == 6

    def test_cmd_pr_merged(self) -> None:
        assert cmd_delivery_id(CmdPRMerged(7, 1)) == 7

    def test_cmd_issue_assigned(self) -> None:
        assert cmd_delivery_id(CmdIssueAssigned(8, 1, "u")) == 8

    def test_cmd_review_submitted(self) -> None:
        assert cmd_delivery_id(CmdReviewSubmitted(9, 1, 1, "u")) == 9


# ---------------------------------------------------------------------------
# evt_delivery_id: accessor extracts the right id from each constructor
# ---------------------------------------------------------------------------


class TestEvtDeliveryId:
    def test_evt_review_comment(self) -> None:
        assert evt_delivery_id(EvtReviewComment(10, 1, 1, "u", False)) == 10

    def test_evt_issue_comment(self) -> None:
        assert evt_delivery_id(EvtIssueComment(20, 1, 1, "u", False)) == 20

    def test_evt_ci_failure(self) -> None:
        assert evt_delivery_id(EvtCIFailure(30, "c", CIFailure(), [])) == 30

    def test_evt_pr_merged(self) -> None:
        assert evt_delivery_id(EvtPRMerged(40, 1)) == 40

    def test_evt_issue_assigned(self) -> None:
        assert evt_delivery_id(EvtIssueAssigned(50, 1, "u")) == 50

    def test_evt_review_submitted(self) -> None:
        assert evt_delivery_id(EvtReviewSubmitted(60, 1, 1, "u")) == 60
