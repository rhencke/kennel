"""GitHub CLI wrappers — all gh subprocess calls in one place."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import requests as _requests

log = logging.getLogger(__name__)

_HTTP_TIMEOUT: int = 30  # seconds for all outbound GitHub HTTP requests

# Retry schedule for transient GitHub failures on idempotent GETs (#664).
# Delays in seconds between successive attempts.  Total retry budget is
# ~14s of wall clock on top of the per-request _HTTP_TIMEOUT.  Only applied
# to read-only paths (GET); mutations stay fail-fast.
_GET_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0, 10.0)
_RETRYABLE_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})


class _TimeoutSession(_requests.Session):
    """requests.Session that applies _HTTP_TIMEOUT to every request by default."""

    def request(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, method: str | bytes, url: str | bytes, **kwargs: Any
    ) -> _requests.Response:  # type: ignore[override]
        kwargs.setdefault("timeout", _HTTP_TIMEOUT)
        return super().request(method, url, **kwargs)


class GraphQLError(Exception):
    """Raised when a GitHub GraphQL response contains an errors field."""

    def __init__(self, errors: list[Any]) -> None:
        super().__init__(f"GraphQL errors: {errors}")
        self.errors = errors


def _auto_merge_unavailable(exc: GraphQLError) -> bool:
    """True when *exc* is a GraphQL ``enablePullRequestAutoMerge`` failure
    caused by the repository having auto-merge disabled.

    GitHub returns ``{'type': 'UNPROCESSABLE', 'message': 'Pull request Auto
    merge is not allowed for this repository', ...}``.  Callers treat that
    as a \"fall back to immediate merge\" signal rather than a crash (fix
    for #643).
    """
    for err in exc.errors:
        if not isinstance(err, dict):
            continue
        if err.get("type") != "UNPROCESSABLE":
            continue
        message = err.get("message") or ""
        if "Auto merge is not allowed" in message:
            return True
    return False


def _gh_token(
    runner: Any = subprocess.run,
    environ: Any = os.environ,
) -> str:
    """Return a GitHub token from env or the gh CLI.

    Raises ``RuntimeError`` on nonzero exit (e.g. not logged in).
    ``FileNotFoundError`` propagates if the gh CLI is not installed.
    ``subprocess.TimeoutExpired`` propagates if the CLI hangs.
    """
    token = environ.get("GITHUB_TOKEN", "")
    if not token:
        result = runner(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"gh auth token failed (exit {result.returncode}) — run `gh auth login`"
            )
        token = result.stdout.strip()
    return token


class GitHub:
    """GitHub client: requests-based REST and GraphQL API calls."""

    BASE = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        session: _requests.Session | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s = session if session is not None else _TimeoutSession()
        self._s.headers.update(
            {
                "Authorization": f"Bearer {token if token is not None else _gh_token()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self._sleep = sleeper

    def _retryable_get(self, url: str) -> _requests.Response:
        """GET *url* with retry on transient upstream failures (#664).

        Retries idempotent GETs on 5xx status codes and on
        ``ConnectionError`` / ``Timeout`` from ``requests``.  Mutation paths
        (POST/PATCH/PUT) never call this — they stay fail-fast.
        """
        last_exc: Exception | None = None
        for attempt in range(len(_GET_RETRY_DELAYS) + 1):
            try:
                resp = self._s.get(url)
                if resp.status_code in _RETRYABLE_STATUS:
                    last_exc = _requests.HTTPError(
                        f"{resp.status_code} {resp.reason} for url: {url}",
                        response=resp,
                    )
                else:
                    resp.raise_for_status()
                    return resp
            except (_requests.ConnectionError, _requests.Timeout) as exc:
                last_exc = exc
            if attempt < len(_GET_RETRY_DELAYS):
                delay = _GET_RETRY_DELAYS[attempt]
                log.warning(
                    "GitHub GET %s failed (%s) — retrying in %.1fs",
                    url,
                    last_exc,
                    delay,
                )
                self._sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _get(self, path: str) -> Any:
        return self._retryable_get(f"{self.BASE}{path}").json()

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
        data = resp.json()
        if "errors" in data:
            raise GraphQLError(data["errors"])
        return data

    def _graphql_paginate(
        self,
        query: str,
        connection_path: tuple[str, ...],
        **variables: Any,
    ) -> Iterator[Any]:
        """Yield every node from a paginated GraphQL connection.

        The *query* must accept a ``$cursor: String`` variable and request
        both ``nodes { ... }`` and ``pageInfo { endCursor hasNextPage }``
        on the connection of interest.

        *connection_path* names the connection's position in the response:
        e.g. ``("repository", "issues")`` walks to
        ``data.repository.issues`` and yields each node from every page.

        Callers pass the initial variables (other than cursor) as keyword
        arguments.  The helper takes care of supplying ``cursor`` on
        subsequent requests.
        """
        cursor: str | None = None
        while True:
            data = self._graphql(query, cursor=cursor, **variables)
            node = data["data"]
            for key in connection_path:
                node = node[key]
            yield from node["nodes"]
            page = node["pageInfo"]
            if not page["hasNextPage"]:
                return
            cursor = page["endCursor"]

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

    def edit_review_comment(self, repo: str, comment_id: int | str, body: str) -> None:
        """Edit the body of an existing inline review comment."""
        self._patch(f"/repos/{repo}/pulls/comments/{comment_id}", body=body)

    def get_pull_comments(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all inline review comments on a pull request."""
        return list(self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/comments"))

    def get_pull_comment(
        self, repo: str, comment_id: int | str
    ) -> dict[str, Any] | None:
        """Return one inline review comment by id, or None if it no longer exists."""
        try:
            return self._get(f"/repos/{repo}/pulls/comments/{comment_id}")
        except _requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

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

    def fetch_comment_thread(
        self, repo: str, pr: int | str, comment_id: int
    ) -> list[dict[str, Any]]:
        """Return all comments in the review thread containing comment_id.

        Returns [{author, body}, ...] in posting order, empty on error.
        """
        try:
            raw = self.get_pull_comments(repo, pr)
        except Exception:
            log.exception("failed to fetch comment thread for %s#%s", repo, pr)
            return []

        by_id = {c["id"]: c for c in raw}
        comment = by_id.get(comment_id)
        if comment is None:
            return []
        root_id = comment.get("in_reply_to_id") or comment_id

        return [
            {
                "id": c["id"],
                "author": c.get("user", {}).get("login", ""),
                "body": c.get("body", ""),
            }
            for c in raw
            if c["id"] == root_id or c.get("in_reply_to_id") == root_id
        ]

    def get_review_comments(
        self, repo: str, pr: int | str, review_id: int | str
    ) -> list[tuple[int, str]]:
        """Return list of (comment_id, body) pairs from a review."""
        url = f"{self.BASE}/repos/{repo}/pulls/{pr}/reviews/{review_id}/comments"
        return [(c["id"], c.get("body", "")) for c in self._paginate(url)]

    def _paginate(self, url: str) -> Iterator[Any]:
        """Yield each item from all pages of a paginated GitHub API endpoint.

        Each page request is retried on transient 5xx / network failures
        (see ``_retryable_get``).  Pagination is a read-only operation so
        retry is always safe.
        """
        current: str | None = url
        while current:
            resp = self._retryable_get(current)
            yield from resp.json()
            link = resp.headers.get("Link", "")
            current = next(
                (
                    part.split(";")[0].strip().strip("<>")
                    for part in link.split(",")
                    if 'rel="next"' in part
                ),
                None,
            )

    def find_pr(
        self, repo: str, issue_number: int | str, user: str
    ) -> dict[str, Any] | None:
        """Find a PR linked to issue_number by user, or None.

        Queries the issue timeline via GraphQL for CrossReferencedEvent (PRs
        with a closing keyword like "closes #N" in their body) and
        ConnectedEvent (PRs manually linked via the Development sidebar).
        DisconnectedEvent removes sidebar-linked PRs that were later unlinked.
        Returns the first open PR found in timeline order.
        """
        owner, name = repo.split("/", 1)
        _CLOSING = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
        pattern = re.compile(rf"(?i)\b{_CLOSING}\s+#{issue_number}\b")
        query = (
            "query($owner:String!,$repo:String!,$number:Int!,$cursor:String){"
            "repository(owner:$owner,name:$repo){"
            "issue(number:$number){"
            "timelineItems("
            "first:100,"
            "itemTypes:[CROSS_REFERENCED_EVENT,CONNECTED_EVENT,DISCONNECTED_EVENT],"
            "after:$cursor"
            "){"
            "pageInfo{hasNextPage endCursor}"
            "nodes{__typename"
            "...on CrossReferencedEvent{source{__typename"
            " ...on PullRequest{number headRefName state title body author{login}}}}"
            "...on ConnectedEvent{subject{__typename"
            " ...on PullRequest{number headRefName state author{login}}}}"
            "...on DisconnectedEvent{subject{__typename"
            " ...on PullRequest{number}}}"
            "}}}}}"
        )
        keyword_prs: set[int] = set()
        sidebar_prs: set[int] = set()
        pr_cache: dict[int, dict[str, Any]] = {}
        cursor: str | None = None
        while True:
            data = self._graphql(
                query,
                owner=owner,
                repo=name,
                number=int(issue_number),
                cursor=cursor,
            )
            items = data["data"]["repository"]["issue"]["timelineItems"]
            if not items:
                return None
            for node in items["nodes"]:
                typename = node["__typename"]
                if typename == "CrossReferencedEvent":
                    pr = node.get("source") or {}
                    if pr.get("__typename") != "PullRequest":
                        continue
                    if pr.get("author", {}).get("login") != user:
                        continue
                    body = pr.get("body", "") or ""
                    title = pr.get("title", "") or ""
                    if not (pattern.search(body) or pattern.search(title)):
                        continue
                    pr_cache.setdefault(pr["number"], pr)
                    keyword_prs.add(pr["number"])
                elif typename == "ConnectedEvent":
                    pr = node.get("subject") or {}
                    if pr.get("__typename") != "PullRequest":
                        continue
                    if pr.get("author", {}).get("login") != user:
                        continue
                    pr_cache.setdefault(pr["number"], pr)
                    sidebar_prs.add(pr["number"])
                elif typename == "DisconnectedEvent":
                    pr = node.get("subject") or {}
                    if pr.get("__typename") == "PullRequest":
                        sidebar_prs.discard(pr["number"])
            page_info = items["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]
        eligible = keyword_prs | sidebar_prs
        for pr_num, pr in pr_cache.items():
            if pr_num not in eligible or pr.get("state") != "OPEN":
                continue
            return {
                "number": pr["number"],
                "headRefName": pr["headRefName"],
                "state": "OPEN",
                "author": {"login": pr["author"]["login"]},
            }
        return None

    def comment_issue(self, repo: str, number: int | str, body: str) -> None:
        """Post a comment on an issue."""
        self._post(f"/repos/{repo}/issues/{number}/comments", body=body)

    def delete_issue_comment(self, repo: str, comment_id: int | str) -> None:
        """Delete an issue/PR top-level comment by id.

        Used by the worker's leak-cleanup path to remove improvised
        top-level PR comments fido sometimes posts during a task turn
        when it can't make progress (see #669).
        """
        resp = self._s.delete(f"{self.BASE}/repos/{repo}/issues/comments/{comment_id}")
        resp.raise_for_status()

    def get_user(self) -> str:
        """Return the authenticated GitHub username."""
        data = self._get("/user")
        return data["login"]

    def get_collaborators(self, repo: str) -> list[str]:
        """Return logins of collaborators with write+ permission on *repo*.

        Filters to users whose permission level is ``admin``, ``maintain``, or
        ``push`` (GitHub's ``write`` equivalent).  Preserves the order returned
        by the API so callers can use ``[0]`` as a stable "primary reviewer".
        """
        result: list[str] = []
        for user in self._paginate(f"{self.BASE}/repos/{repo}/collaborators"):
            perm = user.get("role_name") or ""
            if perm in ("admin", "maintain", "write"):
                login = user.get("login")
                if login:
                    result.append(login)
        return result

    def get_repo_info(
        self,
        cwd: Path | str | None = None,
        runner: Any = subprocess.run,
    ) -> str:
        """Return 'owner/repo' for the repo at cwd, parsed from the git remote.

        Raises ``subprocess.CalledProcessError`` on nonzero exit (e.g. not a
        git repo or no ``origin`` remote).  ``FileNotFoundError`` propagates
        if git is not installed.  Raises ``ValueError`` if the remote URL is
        not a recognised GitHub format.
        """
        result = runner(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
            check=True,
        )
        url = result.stdout.strip().removesuffix(".git")
        if url.startswith("https://github.com/"):
            return url[len("https://github.com/") :]
        if url.startswith("git@github.com:"):
            return url[len("git@github.com:") :]
        raise ValueError(f"Cannot parse GitHub remote URL: {url!r}")

    def get_default_branch(
        self,
        cwd: Path | str | None = None,
        runner: Any = subprocess.run,
    ) -> str:
        """Return the default branch for the repo at cwd."""
        repo = self.get_repo_info(cwd=cwd, runner=runner)
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

    _ISSUE_NODE_FIELDS = (
        "number title createdAt "
        "milestone{title} "
        "assignees(first:20){nodes{login}} "
        "parent{number}"
    )

    def find_issues(self, owner: str, repo: str, login: str) -> list[dict[str, Any]]:
        """Return open issues assigned to *login* (oldest first).

        Each node carries ``number``, ``title``, ``createdAt``,
        ``milestone.title``, ``assignees.nodes[].login``, ``parent.number``
        (or None), and a paginated-and-hydrated ``subIssues.nodes`` list
        in GitHub rank order.
        """
        issue_fields = self._ISSUE_NODE_FIELDS
        query = (
            "query($owner:String!,$repo:String!,$login:String!,$cursor:String){"
            "repository(owner:$owner,name:$repo){"
            "issues(first:50,after:$cursor,states:[OPEN],"
            "filterBy:{assignee:$login},"
            "orderBy:{field:CREATED_AT,direction:ASC}){"
            f"nodes{{{issue_fields} state "
            f"subIssues(first:50){{nodes{{state {issue_fields}}} "
            "pageInfo{endCursor hasNextPage}}}"
            "pageInfo{endCursor hasNextPage}"
            "}}}"
        )
        issues: list[dict[str, Any]] = []
        for node in self._graphql_paginate(
            query,
            ("repository", "issues"),
            owner=owner,
            repo=repo,
            login=login,
        ):
            if node.get("subIssues", {}).get("pageInfo", {}).get("hasNextPage"):
                node["subIssues"]["nodes"] = list(
                    self.get_sub_issues(owner, repo, node["number"])
                )
            issues.append(node)
        return issues

    def find_all_open_issues(self, owner: str, repo: str) -> list[dict[str, Any]]:
        """Return all open issues in the repo (oldest first).

        Each node carries ``number``, ``title``, ``createdAt``, ``state``,
        ``milestone.title``, ``assignees.nodes[].login``, ``parent.number``
        (or None), and a ``subIssues.nodes`` list (up to 100 items — the
        GitHub maximum, so no sub-issue pagination is needed).

        This single paginated call is the basis for the in-memory issue tree
        used by the picker: one or two API calls total regardless of how many
        issues the repo has.
        """
        issue_fields = self._ISSUE_NODE_FIELDS
        query = (
            "query($owner:String!,$repo:String!,$cursor:String){"
            "repository(owner:$owner,name:$repo){"
            "issues(first:50,after:$cursor,states:[OPEN],"
            "orderBy:{field:CREATED_AT,direction:ASC}){"
            f"nodes{{{issue_fields} state "
            f"subIssues(first:100){{nodes{{state {issue_fields}}}}}"
            "}"
            "pageInfo{endCursor hasNextPage}"
            "}}}"
        )
        issues: list[dict[str, Any]] = []
        for node in self._graphql_paginate(
            query,
            ("repository", "issues"),
            owner=owner,
            repo=repo,
        ):
            issues.append(node)
        return issues

    def get_issue_node(self, owner: str, repo: str, number: int) -> dict[str, Any]:
        """Return one issue in the shape used by :meth:`find_issues`.

        Used by the picker's upward walk: call with any issue number and
        get back the same dict shape so descent code can keep walking.
        """
        issue_fields = self._ISSUE_NODE_FIELDS
        query = (
            "query($owner:String!,$repo:String!,$number:Int!){"
            "repository(owner:$owner,name:$repo){"
            "issue(number:$number){"
            f"state {issue_fields} "
            f"subIssues(first:50){{nodes{{state {issue_fields}}} "
            "pageInfo{endCursor hasNextPage}}"
            "}}}"
        )
        data = self._graphql(query, owner=owner, repo=repo, number=number)
        node = data["data"]["repository"]["issue"]
        if node.get("subIssues", {}).get("pageInfo", {}).get("hasNextPage"):
            node["subIssues"]["nodes"] = list(self.get_sub_issues(owner, repo, number))
        return node

    def get_sub_issues(
        self, owner: str, repo: str, number: int
    ) -> Iterator[dict[str, Any]]:
        """Yield the direct sub-issues of *number* in GitHub rank order.

        Each node has the same shape as a node from :meth:`find_issues`
        (``number``, ``title``, ``state``, ``createdAt``, ``milestone.title``,
        ``assignees.nodes[].login``).
        """
        issue_fields = self._ISSUE_NODE_FIELDS
        query = (
            "query($owner:String!,$repo:String!,$number:Int!,$cursor:String){"
            "repository(owner:$owner,name:$repo){"
            "issue(number:$number){"
            "subIssues(first:50,after:$cursor){"
            f"nodes{{state {issue_fields}}} "
            "pageInfo{endCursor hasNextPage}"
            "}}}}"
        )
        yield from self._graphql_paginate(
            query,
            ("repository", "issue", "subIssues"),
            owner=owner,
            repo=repo,
            number=number,
        )

    def add_assignee(self, repo: str, number: int | str, login: str) -> None:
        """Assign *login* to issue *number* in *repo*.

        Uses the REST endpoint so assignment is additive — existing
        assignees are preserved.
        """
        self._post(f"/repos/{repo}/issues/{number}/assignees", assignees=[login])

    def view_issue(self, repo: str, number: int | str) -> dict[str, Any]:
        """Return issue data (state, title, body, created_at)."""
        data = self._get(f"/repos/{repo}/issues/{number}")
        return {
            "state": data["state"].upper(),
            "title": data["title"],
            "body": data["body"] or "",
            "created_at": data.get("created_at", ""),
        }

    def get_issue_comments(self, repo: str, number: int | str) -> list[dict[str, Any]]:
        """Return all comments on an issue."""
        return list(
            self._paginate(f"{self.BASE}/repos/{repo}/issues/{number}/comments")
        )

    def get_issue_comment(
        self, repo: str, comment_id: int | str
    ) -> dict[str, Any] | None:
        """Return one issue comment by id, or None if it no longer exists."""
        try:
            return self._get(f"/repos/{repo}/issues/comments/{comment_id}")
        except _requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise

    def get_issue_events(self, repo: str, number: int | str) -> list[dict[str, Any]]:
        """Return all events on an issue."""
        return list(self._paginate(f"{self.BASE}/repos/{repo}/issues/{number}/events"))

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

    def get_pr_body(self, repo: str, pr: int | str) -> str:
        """Return just the PR body text."""
        data = self._get(f"/repos/{repo}/pulls/{pr}")
        return data.get("body") or ""

    def add_pr_reviewers(self, repo: str, pr: int | str, reviewers: list[str]) -> None:
        """Request review from one or more users on a PR.

        GitHub's API accepts a list in a single call, so there is no
        singular form — always pass the full set of collaborators to
        request from.
        """
        if not reviewers:
            return
        self._post(
            f"/repos/{repo}/pulls/{pr}/requested_reviewers",
            reviewers=list(reviewers),
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
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        node_id = pr_data["node_id"]
        query = (
            "mutation($prId:ID!){"
            "markPullRequestReadyForReview(input:{pullRequestId:$prId})"
            "{pullRequest{isDraft}}}"
        )
        self._graphql(query, prId=node_id)

    def pr_merge(
        self, repo: str, pr: int | str, squash: bool = True, auto: bool = False
    ) -> None:
        """Merge a PR.

        If the PR is already merged, logs and returns success rather than
        raising.  This handles the race where kennel self-restarts after a
        PR merge and the worker resumes with stale state thinking the PR
        still needs merging.
        """
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        if pr_data.get("merged"):
            log.info("PR %s/#%s already merged — skipping", repo, pr)
            return
        if auto and self._try_enable_auto_merge(repo, pr, pr_data, squash):
            return
        merge_method = "squash" if squash else "merge"
        try:
            self._put(
                f"/repos/{repo}/pulls/{pr}/merge",
                merge_method=merge_method,
            )
        except _requests.HTTPError as e:
            # Re-check: if the PR was merged between our initial get_pr and
            # the merge call (race with webhook-triggered self-restart on
            # another process), treat 405 as success.
            if e.response is not None and e.response.status_code == 405:
                recheck = self._get(f"/repos/{repo}/pulls/{pr}")
                if recheck.get("merged"):
                    log.info(
                        "PR %s/#%s merged concurrently — treating 405 as success",
                        repo,
                        pr,
                    )
                    return
            raise

    def _try_enable_auto_merge(
        self,
        repo: str,
        pr: int | str,
        pr_data: dict[str, Any],
        squash: bool,
    ) -> bool:
        """Try to enable auto-merge on *pr*.

        Returns True on success.  Returns False when the repository has
        auto-merge disabled (GraphQL UNPROCESSABLE), so the caller can
        fall back to an immediate REST merge.  Re-raises any other
        :class:`GraphQLError` — callers are not equipped to continue past
        a genuine protocol failure.
        """
        node_id = pr_data["node_id"]
        merge_method = "SQUASH" if squash else "MERGE"
        query = (
            "mutation($prId:ID!,$mergeMethod:PullRequestMergeMethod!){"
            "enablePullRequestAutoMerge(input:{pullRequestId:$prId,"
            "mergeMethod:$mergeMethod})"
            "{pullRequest{autoMergeRequest{mergeMethod}}}}"
        )
        try:
            self._graphql(query, prId=node_id, mergeMethod=merge_method)
            return True
        except GraphQLError as exc:
            if not _auto_merge_unavailable(exc):
                raise
            log.info(
                "PR %s/#%s: auto-merge disabled on repo — "
                "falling back to immediate %s merge",
                repo,
                pr,
                "squash" if squash else "merge",
            )
            return False

    def get_pr(self, repo: str, pr: int | str) -> dict[str, Any]:
        """Return PR data (reviews, isDraft, mergeStateStatus, body, commits)."""
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        reviews_data = list(
            self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/reviews")
        )
        commits_data = list(
            self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/commits")
        )
        reviews = [
            {
                "author": {"login": r["user"]["login"]},
                "state": r["state"],
                "submittedAt": r["submitted_at"],
                "body": r.get("body", "") or "",
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
        reviews_data = list(
            self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/reviews")
        )
        commits_data = list(
            self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/commits")
        )
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
        requested_reviewers = [
            u["login"] for u in pr_data.get("requested_reviewers", [])
        ]
        return {
            "reviews": reviews,
            "commits": commits,
            "isDraft": pr_data["draft"],
            "requestedReviewers": requested_reviewers,
        }

    def get_review_threads(
        self, owner: str, repo: str, pr: int | str
    ) -> list[dict[str, Any]]:
        """Return review-thread nodes for a PR."""
        query = (
            "query($owner:String!,$repo:String!,$pr:Int!){"
            "repository(owner:$owner,name:$repo){"
            "pullRequest(number:$pr){"
            "reviewThreads(first:100){"
            "nodes{id isResolved"
            " comments(first:50){nodes{author{login} body url createdAt databaseId}}"
            "}}}}}"
        )
        data = self._graphql(query, owner=owner, repo=repo, pr=int(pr))
        return data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]

    def resolve_thread(self, thread_id: str) -> None:
        """Resolve a review thread via GraphQL mutation."""
        query = (
            "mutation($id:ID!)"
            "{resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}"
        )
        self._graphql(query, id=thread_id)

    def is_thread_resolved_for_comment(
        self, repo: str, pr: int | str, comment_id: int
    ) -> bool:
        """Return True if the review thread containing *comment_id* is
        already resolved on GitHub.

        Used by webhook task creation to skip queuing late-arriving thread
        tasks for threads fido has already auto-resolved (closes #520 race
        / #521 redo loop).  Returns False when the comment isn't found in
        any thread (treat as "go ahead and queue").
        """
        owner, name = repo.split("/", 1)
        for node in self.get_review_threads(owner, name, pr):
            for c in node.get("comments", {}).get("nodes", []):
                if c.get("databaseId") == comment_id:
                    return bool(node.get("isResolved"))
        return False

    def get_required_checks(self, repo: str, branch: str) -> list[str]:
        """Return required status check names for branch, or [] if unprotected."""
        try:
            data = self._get(f"/repos/{repo}/branches/{branch}/protection")
        except _requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return []
            raise
        rsc = data.get("required_status_checks") or {}
        return [c["context"] for c in rsc.get("checks", [])]

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
