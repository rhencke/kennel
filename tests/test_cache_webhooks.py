"""Tests for fido.cache_webhooks.translate (closes #812, #817)."""

from datetime import datetime, timezone

from fido.cache_webhooks import translate


def _sub_issues_payload(
    action: str,
    *,
    parent_number: int = 195,
    sub_number: int = 222,
    parent_updated_at: str = "2026-04-19T01:00:00Z",
    sub_updated_at: str = "2026-04-19T01:00:00Z",
    parent_created_at: str = "2026-04-15T00:00:00Z",
    sub_created_at: str = "2026-04-15T00:00:00Z",
    parent_present: bool = True,
    sub_present: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {"action": action}
    if parent_present:
        payload["parent_issue"] = {
            "number": parent_number,
            "updated_at": parent_updated_at,
            "created_at": parent_created_at,
        }
    if sub_present:
        payload["sub_issue"] = {
            "number": sub_number,
            "updated_at": sub_updated_at,
            "created_at": sub_created_at,
        }
    return payload


def _issue(
    number: int = 7,
    *,
    title: str = "T",
    assignees: list[str] | None = None,
    milestone: str | None = None,
    parent: int | None = None,
    updated_at: str = "2026-04-18T22:00:00Z",
    created_at: str = "2026-04-15T00:00:00Z",
) -> dict[str, object]:
    return {
        "number": number,
        "title": title,
        "assignees": [{"login": a} for a in (assignees or [])],
        "milestone": {"title": milestone} if milestone else None,
        "parent": {"number": parent} if parent is not None else None,
        "updated_at": updated_at,
        "created_at": created_at,
    }


class TestNonIssueEvent:
    def test_pull_request_returns_none(self) -> None:
        assert translate("pull_request", {"action": "closed"}) is None

    def test_issue_comment_returns_none(self) -> None:
        assert translate("issue_comment", {"action": "created"}) is None

    def test_unknown_event_returns_none(self) -> None:
        assert translate("workflow_run", {"action": "completed"}) is None


class TestIssueOpenedReopened:
    def test_opened_carries_full_snapshot(self) -> None:
        result = translate(
            "issues",
            {
                "action": "opened",
                "issue": _issue(7, title="hi", assignees=["fido"], milestone="v1"),
            },
        )
        assert result is not None
        evt_type, payload = result
        assert evt_type == "opened"
        assert payload["issue_number"] == 7
        assert payload["title"] == "hi"
        assert payload["assignees"] == ["fido"]
        assert payload["milestone"] == "v1"
        assert payload["parent"] is None
        assert payload["sub_issues"] == []
        assert isinstance(payload["timestamp"], datetime)
        assert payload["timestamp"].tzinfo is not None

    def test_reopened_emits_reopened_type(self) -> None:
        result = translate("issues", {"action": "reopened", "issue": _issue(7)})
        assert result is not None
        assert result[0] == "reopened"

    def test_opened_with_parent(self) -> None:
        result = translate(
            "issues",
            {"action": "opened", "issue": _issue(7, parent=99)},
        )
        assert result is not None
        assert result[1]["parent"] == 99

    def test_opened_without_issue_returns_none(self) -> None:
        assert translate("issues", {"action": "opened"}) is None


class TestIssueClosedFamily:
    def test_closed(self) -> None:
        result = translate("issues", {"action": "closed", "issue": _issue(7)})
        assert result is not None
        assert result[0] == "closed"
        assert result[1]["issue_number"] == 7

    def test_transferred_treated_as_closed(self) -> None:
        result = translate("issues", {"action": "transferred", "issue": _issue(7)})
        assert result is not None
        assert result[0] == "closed"

    def test_deleted_treated_as_closed(self) -> None:
        result = translate("issues", {"action": "deleted", "issue": _issue(7)})
        assert result is not None
        assert result[0] == "closed"


class TestAssignedUnassigned:
    def test_assigned(self) -> None:
        result = translate(
            "issues",
            {
                "action": "assigned",
                "issue": _issue(7),
                "assignee": {"login": "alice"},
            },
        )
        assert result is not None
        assert result[0] == "assigned"
        assert result[1]["login"] == "alice"
        assert result[1]["issue_number"] == 7

    def test_unassigned(self) -> None:
        result = translate(
            "issues",
            {
                "action": "unassigned",
                "issue": _issue(7),
                "assignee": {"login": "bob"},
            },
        )
        assert result is not None
        assert result[0] == "unassigned"
        assert result[1]["login"] == "bob"

    def test_assigned_without_assignee_returns_none(self) -> None:
        assert translate("issues", {"action": "assigned", "issue": _issue(7)}) is None

    def test_unassigned_without_assignee_returns_none(self) -> None:
        assert translate("issues", {"action": "unassigned", "issue": _issue(7)}) is None


class TestMilestoned:
    def test_milestoned_carries_milestone(self) -> None:
        result = translate(
            "issues",
            {"action": "milestoned", "issue": _issue(7, milestone="v2")},
        )
        assert result is not None
        assert result[0] == "milestoned"
        assert result[1]["milestone"] == "v2"

    def test_demilestoned_clears_to_none(self) -> None:
        result = translate(
            "issues",
            {"action": "demilestoned", "issue": _issue(7, milestone=None)},
        )
        assert result is not None
        assert result[0] == "milestoned"  # collapsed onto same handler
        assert result[1]["milestone"] is None


class TestEdited:
    def test_title_change_emits_event(self) -> None:
        result = translate(
            "issues",
            {
                "action": "edited",
                "issue": _issue(7, title="new"),
                "changes": {"title": {"from": "old"}},
            },
        )
        assert result is not None
        assert result[0] == "edited_title"
        assert result[1]["title"] == "new"

    def test_body_only_change_returns_none(self) -> None:
        result = translate(
            "issues",
            {
                "action": "edited",
                "issue": _issue(7),
                "changes": {"body": {"from": "old body"}},
            },
        )
        assert result is None

    def test_no_changes_returns_none(self) -> None:
        result = translate("issues", {"action": "edited", "issue": _issue(7)})
        assert result is None


class TestSubIssue:
    def test_sub_issue_added(self) -> None:
        result = translate(
            "issues",
            {
                "action": "sub_issue_added",
                "issue": _issue(7),
                "sub_issue": {"number": 99},
            },
        )
        assert result is not None
        assert result[0] == "sub_issue_added"
        assert result[1]["issue_number"] == 7  # parent
        assert result[1]["child"] == 99

    def test_sub_issue_removed(self) -> None:
        result = translate(
            "issues",
            {
                "action": "sub_issue_removed",
                "issue": _issue(7),
                "sub_issue": {"number": 99},
            },
        )
        assert result is not None
        assert result[0] == "sub_issue_removed"
        assert result[1]["child"] == 99

    def test_sub_issue_added_without_child_returns_none(self) -> None:
        assert (
            translate(
                "issues",
                {"action": "sub_issue_added", "issue": _issue(7)},
            )
            is None
        )

    def test_sub_issue_removed_without_child_returns_none(self) -> None:
        assert (
            translate(
                "issues",
                {"action": "sub_issue_removed", "issue": _issue(7)},
            )
            is None
        )


class TestUnsupportedAction:
    def test_labeled_returns_none(self) -> None:
        # Labels aren't tracked by the picker.
        assert translate("issues", {"action": "labeled", "issue": _issue(7)}) is None

    def test_pinned_returns_none(self) -> None:
        assert translate("issues", {"action": "pinned", "issue": _issue(7)}) is None


class TestEdgeCases:
    def test_issue_without_number_returns_none(self) -> None:
        result = translate("issues", {"action": "assigned", "issue": {"title": "x"}})
        assert result is None

    def test_missing_timestamps_falls_back_to_now(self) -> None:
        result = translate(
            "issues",
            {
                "action": "opened",
                "issue": {"number": 7, "title": "x"},
            },
        )
        assert result is not None
        assert isinstance(result[1]["timestamp"], datetime)
        assert result[1]["timestamp"].tzinfo is timezone.utc

    def test_issue_milestone_with_non_string_title(self) -> None:
        # GitHub sometimes returns null in surprising places.
        result = translate(
            "issues",
            {
                "action": "milestoned",
                "issue": {
                    "number": 7,
                    "title": "x",
                    "milestone": {"title": None},
                    "updated_at": "2026-04-18T22:00:00Z",
                    "created_at": "2026-04-15T00:00:00Z",
                },
            },
        )
        assert result is not None
        assert result[1]["milestone"] is None


# ── sub_issues event family (closes #817) ────────────────────────────────────


class TestSubIssuesEvent:
    """Translate the dedicated ``sub_issues`` webhook event."""

    def test_sub_issue_added_maps_to_cache_event(self) -> None:
        result = translate(
            "sub_issues",
            _sub_issues_payload("sub_issue_added", parent_number=195, sub_number=222),
        )
        assert result is not None
        cache_event, payload = result
        assert cache_event == "sub_issue_added"
        assert payload["issue_number"] == 195
        assert payload["child"] == 222

    def test_sub_issue_removed_maps_to_cache_event(self) -> None:
        result = translate(
            "sub_issues",
            _sub_issues_payload("sub_issue_removed", parent_number=195, sub_number=222),
        )
        assert result is not None
        assert result[0] == "sub_issue_removed"
        assert result[1]["issue_number"] == 195
        assert result[1]["child"] == 222

    def test_parent_issue_added_maps_to_sub_issue_added_keyed_by_parent(self) -> None:
        """Child-side notification maps to the same parent-keyed cache event."""
        result = translate(
            "sub_issues",
            _sub_issues_payload(
                "parent_issue_added", parent_number=195, sub_number=222
            ),
        )
        assert result is not None
        assert result[0] == "sub_issue_added"
        assert result[1]["issue_number"] == 195
        assert result[1]["child"] == 222

    def test_parent_issue_removed_maps_to_sub_issue_removed(self) -> None:
        result = translate(
            "sub_issues",
            _sub_issues_payload(
                "parent_issue_removed", parent_number=195, sub_number=222
            ),
        )
        assert result is not None
        assert result[0] == "sub_issue_removed"
        assert result[1]["issue_number"] == 195
        assert result[1]["child"] == 222

    def test_returns_none_when_parent_missing(self) -> None:
        assert (
            translate(
                "sub_issues",
                _sub_issues_payload("sub_issue_added", parent_present=False),
            )
            is None
        )

    def test_returns_none_when_sub_missing(self) -> None:
        assert (
            translate(
                "sub_issues",
                _sub_issues_payload("sub_issue_added", sub_present=False),
            )
            is None
        )

    def test_returns_none_for_unknown_action(self) -> None:
        assert translate("sub_issues", _sub_issues_payload("nonsense")) is None

    def test_timestamp_is_now_not_payload_updated_at(self) -> None:
        """Sub-issue relationship mutations don't bump either end's
        ``updated_at``; using that field caused the cache to drop legit
        mutations as stale (#819).  Translator must use ``now()`` so the
        cache's last_applied_at monotonicity check accepts them."""
        payload: dict[str, object] = {
            "action": "sub_issue_added",
            "parent_issue": {
                "number": 1,
                "updated_at": "2020-01-01T00:00:00Z",
                "created_at": "2020-01-01T00:00:00Z",
            },
            "sub_issue": {
                "number": 2,
                "updated_at": "2020-01-01T00:00:00Z",
                "created_at": "2020-01-01T00:00:00Z",
            },
        }
        before = datetime.now(tz=timezone.utc)
        result = translate("sub_issues", payload)
        after = datetime.now(tz=timezone.utc)
        assert result is not None
        ts = result[1]["timestamp"]
        assert isinstance(ts, datetime)
        assert before <= ts <= after

    def test_timestamp_works_without_any_issue_timestamps(self) -> None:
        """Payload with no updated_at / created_at on either end must
        still produce a usable timestamp (closes #819)."""
        payload: dict[str, object] = {
            "action": "sub_issue_removed",
            "parent_issue": {"number": 1},
            "sub_issue": {"number": 2},
        }
        result = translate("sub_issues", payload)
        assert result is not None
        assert isinstance(result[1]["timestamp"], datetime)
        assert result[1]["timestamp"].tzinfo is not None
