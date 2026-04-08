"""GitHub CLI wrappers — all gh subprocess calls in one place."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests as _requests

log = logging.getLogger(__name__)

# ── Requests-based GitHub API client ─────────────────────────────────────────


def _gh_token() -> str:
    """Return a GitHub token from env or the gh CLI."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        token = result.stdout.strip()
    return token


class GH:
    """GitHub client: requests-based REST and GraphQL API calls."""

    BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._s = _requests.Session()
        self._s.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _get(self, path: str) -> Any:
        resp = self._s.get(f"{self.BASE}{path}")
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, **payload: Any) -> None:
        resp = self._s.post(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()

    def _post_json(self, path: str, **payload: Any) -> Any:
        resp = self._s.post(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, path: str, **payload: Any) -> Any:
        resp = self._s.patch(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, **payload: Any) -> Any:
        resp = self._s.put(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _graphql(self, query: str, **variables: Any) -> Any:
        """Execute a GraphQL query/mutation against the GitHub API."""
        resp = self._s.post(
            f"{self.BASE}/graphql",
            json={"query": query, "variables": variables},
        )
        resp.raise_for_status()
        return resp.json()

    def add_reaction(
        self, repo: str, comment_type: str, comment_id: int | str, content: str
    ) -> None:
        """Add a reaction to a comment. comment_type: 'pulls' or 'issues'."""
        self._post(
            f"/repos/{repo}/{comment_type}/comments/{comment_id}/reactions",
            content=content,
        )

    def reply_to_review_comment(
        self, repo: str, pr: int | str, body: str, in_reply_to: int | str
    ) -> None:
        """Post a reply to an inline review comment."""
        self._post(
            f"/repos/{repo}/pulls/{pr}/comments",
            body=body,
            in_reply_to=int(in_reply_to),
        )

    def get_pull_comments(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all inline review comments on a pull request."""
        return self._get(f"/repos/{repo}/pulls/{pr}/comments")

    def fetch_sibling_threads(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all review-comment threads for a PR as a structured list.

        Each entry is {path, line, comments: [{author, body}, ...]}.
        Root comments (no in_reply_to_id) start a new thread; replies are appended.
        Returns [] on any error.
        """
        try:
            raw = self.get_pull_comments(repo, pr)
        except Exception:
            log.exception("failed to fetch sibling threads for %s#%s", repo, pr)
            return []

        threads: dict[int, dict[str, Any]] = {}
        for c in raw:
            cid = c["id"]
            parent_id = c.get("in_reply_to_id")
            entry = {
                "author": c.get("user", {}).get("login", ""),
                "body": c.get("body", ""),
            }
            if parent_id is None:
                threads[cid] = {
                    "path": c.get("path", ""),
                    "line": c.get("line"),
                    "comments": [entry],
                }
            else:
                root = threads.get(parent_id)
                if root is not None:
                    root["comments"].append(entry)

        return list(threads.values())

    def get_review_comments(
        self, repo: str, pr: int | str, review_id: int | str
    ) -> list[tuple[int, str]]:
        """Return list of (comment_id, body) pairs from a review."""
        data = self._get(f"/repos/{repo}/pulls/{pr}/reviews/{review_id}/comments")
        return [(c["id"], c.get("body", "")) for c in data]

    def find_pr(
        self, repo: str, issue_number: int | str, user: str
    ) -> dict[str, Any] | None:
        """Find the most recent PR with issue_number in body by user."""
        q = quote(f"#{issue_number} in:body repo:{repo} type:pr")
        data = self._get(f"/search/issues?q={q}")
        for item in data.get("items", []):
            if item.get("user", {}).get("login") != user:
                continue
            pr = self._get(f"/repos/{repo}/pulls/{item['number']}")
            state = "MERGED" if pr.get("merged") else pr["state"].upper()
            return {
                "number": pr["number"],
                "headRefName": pr["head"]["ref"],
                "state": state,
                "author": {"login": pr["user"]["login"]},
            }
        return None

    def comment_issue(self, repo: str, number: int | str, body: str) -> None:
        """Post a comment on an issue."""
        self._post(f"/repos/{repo}/issues/{number}/comments", body=body)

    def get_user(self) -> str:
        """Return the authenticated GitHub username."""
        data = self._get("/user")
        return data["login"]

    def get_repo_info(self, cwd: Path | str | None = None) -> str:
        """Return 'owner/repo' for the repo at cwd, parsed from the git remote."""
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        url = result.stdout.strip().removesuffix(".git")
        if url.startswith("https://github.com/"):
            return url[len("https://github.com/") :]
        if url.startswith("git@github.com:"):
            return url[len("git@github.com:") :]
        raise ValueError(f"Cannot parse GitHub remote URL: {url!r}")

    def get_default_branch(self, repo: str) -> str:
        """Return the default branch name for the given repo."""
        data = self._get(f"/repos/{repo}")
        return data["default_branch"]

    def set_user_status(self, msg: str, emoji: str, busy: bool = True) -> None:
        """Set the authenticated user's GitHub status."""
        query = (
            "mutation($msg:String!,$emoji:String!,$busy:Boolean!)"
            "{ changeUserStatus(input: {message: $msg, emoji: $emoji,"
            " limitedAvailability: $busy})"
            "{ status { message emoji indicatesLimitedAvailability } } }"
        )
        self._graphql(query, msg=msg, emoji=emoji, busy=busy)

    def find_issues(self, owner: str, repo: str, login: str) -> list[dict[str, Any]]:
        """Return open issues assigned to login (oldest first) with sub-issue states."""
        query = (
            "query($owner:String!,$repo:String!,$login:String!){"
            "repository(owner:$owner,name:$repo){"
            "issues(first:50,states:[OPEN],filterBy:{assignee:$login},"
            "orderBy:{field:CREATED_AT,direction:ASC}){"
            "nodes{number title subIssues(first:10){nodes{state}}}}}}"
        )
        data = self._graphql(query, owner=owner, repo=repo, login=login)
        return data["data"]["repository"]["issues"]["nodes"]

    def view_issue(self, repo: str, number: int | str) -> dict[str, Any]:
        """Return issue data (state, title, body)."""
        data = self._get(f"/repos/{repo}/issues/{number}")
        return {
            "state": data["state"].upper(),
            "title": data["title"],
            "body": data["body"] or "",
        }

    def get_issue_comments(self, repo: str, number: int | str) -> list[dict[str, Any]]:
        """Return all comments on an issue."""
        return self._get(f"/repos/{repo}/issues/{number}/comments")

    def create_issue(self, repo: str, title: str, body: str) -> str:
        """Create an issue and return its HTML URL."""
        data = self._post_json(f"/repos/{repo}/issues", title=title, body=body)
        return data["html_url"]

    def create_pr(self, repo: str, title: str, body: str, base: str, head: str) -> str:
        """Create a draft PR and return its URL."""
        data = self._post_json(
            f"/repos/{repo}/pulls",
            title=title,
            body=body,
            base=base,
            head=head,
            draft=True,
        )
        return data["html_url"]

    def edit_pr_body(self, repo: str, pr: int | str, body: str) -> None:
        """Edit a PR's body."""
        self._patch(f"/repos/{repo}/pulls/{pr}", body=body)

    def add_pr_reviewer(self, repo: str, pr: int | str, reviewer: str) -> None:
        """Add a reviewer to a PR."""
        self._post(
            f"/repos/{repo}/pulls/{pr}/requested_reviewers",
            reviewers=[reviewer],
        )

    def pr_checks(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return check statuses for a PR."""
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        sha = pr_data["head"]["sha"]
        data = self._get(f"/repos/{repo}/commits/{sha}/check-runs")
        result = []
        for run in data.get("check_runs", []):
            if run["status"] == "completed":
                state = (run["conclusion"] or "").upper()
            else:
                state = run["status"].upper()
            result.append(
                {"name": run["name"], "state": state, "link": run["html_url"]}
            )
        return result

    def pr_ready(self, repo: str, pr: int | str) -> None:
        """Mark a PR ready for review."""
        self._patch(f"/repos/{repo}/pulls/{pr}", draft=False)

    def pr_merge(
        self, repo: str, pr: int | str, squash: bool = True, auto: bool = False
    ) -> None:
        """Merge a PR."""
        if auto:
            pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
            node_id = pr_data["node_id"]
            merge_method = "SQUASH" if squash else "MERGE"
            query = (
                "mutation($prId:ID!,$mergeMethod:PullRequestMergeMethod!){"
                "enablePullRequestAutoMerge(input:{pullRequestId:$prId,"
                "mergeMethod:$mergeMethod})"
                "{pullRequest{autoMergeRequest{mergeMethod}}}}"
            )
            self._graphql(query, prId=node_id, mergeMethod=merge_method)
        else:
            merge_method = "squash" if squash else "merge"
            self._put(
                f"/repos/{repo}/pulls/{pr}/merge",
                merge_method=merge_method,
            )

    def get_pr(self, repo: str, pr: int | str) -> dict[str, Any]:
        """Return PR data (reviews, isDraft, mergeStateStatus, body, commits)."""
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        reviews_data = self._get(f"/repos/{repo}/pulls/{pr}/reviews")
        commits_data = self._get(f"/repos/{repo}/pulls/{pr}/commits")
        reviews = [
            {
                "author": {"login": r["user"]["login"]},
                "state": r["state"],
                "submittedAt": r["submitted_at"],
            }
            for r in reviews_data
        ]
        commits = [
            {
                "messageHeadline": c["commit"]["message"].split("\n")[0],
                "oid": c["sha"],
                "committedDate": c["commit"].get("committer", {}).get("date", ""),
            }
            for c in commits_data
        ]
        return {
            "reviews": reviews,
            "isDraft": pr_data["draft"],
            "mergeStateStatus": (pr_data.get("mergeable_state") or "").upper(),
            "body": pr_data["body"] or "",
            "commits": commits,
        }

    def get_reviews(self, repo: str, pr: int | str) -> dict[str, Any]:
        """Return reviews, commits, and isDraft for a PR."""
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        reviews_data = self._get(f"/repos/{repo}/pulls/{pr}/reviews")
        commits_data = self._get(f"/repos/{repo}/pulls/{pr}/commits")
        reviews = [
            {
                "author": {"login": r["user"]["login"]},
                "state": r["state"],
                "submittedAt": r["submitted_at"],
            }
            for r in reviews_data
        ]
        commits = [
            {
                "committedDate": c["commit"].get("committer", {}).get("date", ""),
            }
            for c in commits_data
        ]
        return {
            "reviews": reviews,
            "commits": commits,
            "isDraft": pr_data["draft"],
        }

    def get_review_threads(
        self, owner: str, repo: str, pr: int | str
    ) -> dict[str, Any]:
        """Return full review-thread data for a PR (GraphQL)."""
        query = (
            "query($owner:String!,$repo:String!,$pr:Int!){"
            "repository(owner:$owner,name:$repo){"
            "pullRequest(number:$pr){"
            "reviewThreads(first:100){"
            "nodes{id isResolved"
            " comments(first:50){nodes{author{login} body url createdAt databaseId}}"
            "}}}}}"
        )
        return self._graphql(query, owner=owner, repo=repo, pr=int(pr))

    def resolve_thread(self, thread_id: str) -> None:
        """Resolve a review thread via GraphQL mutation."""
        query = (
            "mutation($id:ID!)"
            "{resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}"
        )
        self._graphql(query, id=thread_id)

    def get_run_log(self, repo: str, run_id: str | int) -> str:
        """Return the failed log output for a CI run."""
        jobs_data = self._get(f"/repos/{repo}/actions/runs/{run_id}/jobs?filter=latest")
        parts = []
        for job in jobs_data.get("jobs", []):
            if job.get("conclusion") not in ("failure", "timed_out"):
                continue
            resp = self._s.get(
                f"{self.BASE}/repos/{repo}/actions/jobs/{job['id']}/logs"
            )
            resp.raise_for_status()
            parts.append(resp.text)
        return "".join(parts)


@functools.cache
def _get_gh() -> GH:
    """Return the shared GH instance, creating it on first call."""
    return GH(_gh_token())


class GitHub:
    """Facade that stores a single GH client as self._gh and exposes named methods."""

    def __init__(self, token: str | None = None) -> None:
        self._gh = GH(token if token is not None else _gh_token())

    # ── Repo / user context ───────────────────────────────────────────────────

    def get_repo_info(self, cwd: Path | str | None = None) -> str:
        """Return 'owner/repo' for the repo at cwd."""
        return self._gh.get_repo_info(cwd=cwd)

    def get_user(self) -> str:
        """Return the authenticated GitHub username."""
        return self._gh.get_user()

    def get_default_branch(self, cwd: Path | str | None = None) -> str:
        """Return the default branch name for the repo at cwd."""
        repo = self._gh.get_repo_info(cwd=cwd)
        return self._gh.get_default_branch(repo)

    def set_user_status(self, msg: str, emoji: str, busy: bool = True) -> None:
        """Set the authenticated user's GitHub status."""
        self._gh.set_user_status(msg, emoji, busy)

    # ── Issues ────────────────────────────────────────────────────────────────

    def find_issues(self, owner: str, repo: str, login: str) -> list[dict[str, Any]]:
        """Return open issues assigned to login (oldest first) with sub-issue states."""
        return self._gh.find_issues(owner, repo, login)

    def view_issue(self, repo: str, number: int | str) -> dict[str, Any]:
        """Return issue data (state, title, body)."""
        return self._gh.view_issue(repo, number)

    def comment_issue(self, repo: str, number: int | str, body: str) -> None:
        """Post a comment on an issue."""
        self._gh.comment_issue(repo, number, body)

    def get_issue_comments(self, repo: str, number: int | str) -> list[dict[str, Any]]:
        """Return all comments on an issue."""
        return self._gh.get_issue_comments(repo, number)

    def create_issue(self, repo: str, title: str, body: str) -> str:
        """Create an issue and return its HTML URL."""
        return self._gh.create_issue(repo, title, body)

    # ── Pull requests ─────────────────────────────────────────────────────────

    def get_pull_comments(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all inline review comments on a pull request."""
        return self._gh.get_pull_comments(repo, pr)

    def fetch_sibling_threads(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all review-comment threads for a PR as a structured list."""
        return self._gh.fetch_sibling_threads(repo, pr)

    def find_pr(
        self, repo: str, issue_number: int | str, user: str
    ) -> dict[str, Any] | None:
        """Find the most recent PR linked to issue_number authored by user, or None."""
        return self._gh.find_pr(repo, issue_number, user)

    def create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        base: str,
        head: str,
    ) -> str:
        """Create a draft PR and return its URL."""
        return self._gh.create_pr(repo, title, body, base, head)

    def edit_pr_body(self, repo: str, pr: int | str, body: str) -> None:
        """Edit a PR's body."""
        self._gh.edit_pr_body(repo, pr, body)

    def add_pr_reviewer(self, repo: str, pr: int | str, reviewer: str) -> None:
        """Add a reviewer to a PR."""
        self._gh.add_pr_reviewer(repo, pr, reviewer)

    def pr_checks(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return check statuses for a PR."""
        return self._gh.pr_checks(repo, pr)

    def pr_ready(self, repo: str, pr: int | str) -> None:
        """Mark a PR ready for review."""
        self._gh.pr_ready(repo, pr)

    def pr_merge(
        self, repo: str, pr: int | str, squash: bool = True, auto: bool = False
    ) -> None:
        """Merge a PR."""
        self._gh.pr_merge(repo, pr, squash=squash, auto=auto)

    def get_pr(self, repo: str, pr: int | str) -> dict[str, Any]:
        """Return PR data (reviews, isDraft, mergeStateStatus, body, commits)."""
        return self._gh.get_pr(repo, pr)

    def get_reviews(self, repo: str, pr: int | str) -> dict[str, Any]:
        """Return reviews and isDraft for a PR."""
        return self._gh.get_reviews(repo, pr)

    def get_review_comments(
        self, repo: str, pr: int | str, review_id: int | str
    ) -> list[tuple[int, str]]:
        """Return list of (comment_id, body) pairs from a review."""
        return self._gh.get_review_comments(repo, pr, review_id)

    def reply_to_review_comment(
        self,
        repo: str,
        pr: int | str,
        body: str,
        in_reply_to: int | str,
    ) -> None:
        """Post a reply to an inline review comment."""
        self._gh.reply_to_review_comment(repo, pr, body, in_reply_to)

    def add_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        content: str,
    ) -> None:
        """Add a reaction to a comment. comment_type: 'pulls' or 'issues'."""
        self._gh.add_reaction(repo, comment_type, comment_id, content)

    # ── Review threads ────────────────────────────────────────────────────────

    def get_review_threads(
        self, owner: str, repo: str, pr: int | str
    ) -> dict[str, Any]:
        """Return full review-thread data for a PR (GraphQL)."""
        return self._gh.get_review_threads(owner, repo, pr)

    def resolve_thread(self, thread_id: str) -> None:
        """Resolve a review thread via GraphQL mutation."""
        self._gh.resolve_thread(thread_id)

    # ── CI runs ───────────────────────────────────────────────────────────────

    def get_run_log(self, repo: str, run_id: str | int) -> str:
        """Return the failed log output for a CI run."""
        return self._gh.get_run_log(repo, run_id)


@functools.cache
def get_github() -> GitHub:
    """Return the shared GitHub facade instance, creating it on first call."""
    return GitHub()
