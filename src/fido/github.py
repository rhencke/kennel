"""GitHub CLI wrappers — all gh subprocess calls in one place."""

import logging
import os
import re
import subprocess
import threading
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import requests as _requests

from fido.types import ClosedPR, ClosedSubIssue, GitIdentity

log = logging.getLogger(__name__)

_HTTP_TIMEOUT: int = 30  # seconds for all outbound GitHub HTTP requests

# Matches GitHub's closing-keyword syntax in PR body/title text.
# Used to identify closing PRs for already-closed issues, where
# willCloseTarget is always false once the target is closed.
#
# GitHub's full set of closing keywords (all tenses/numbers):
#   close, closes, closed
#   fix, fixes, fixed
#   resolve, resolves, resolved
# Optional colon after the keyword (e.g. "Fixes: #123").
# Optional owner/repo prefix for cross-repo references (e.g. "Fixes owner/repo#123").
# Word boundary \b before keyword prevents false matches like "prefix_closes #N".
_CLOSING_KEYWORD_RE: re.Pattern[str] = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+"
    r"(?:([\w.-]+/[\w.-]+)#|#)(\d+)"
)


def _has_closing_keyword(
    text: str, issue_number: int | str, repo: str | None = None
) -> bool:
    """Return ``True`` when *text* contains a closing keyword referencing *issue_number*.

    For bare ``#N`` references, the match is unconditional.  For cross-repo
    references (``owner/repo#N``), the reference only matches when *repo* is
    provided and equals the prefix (case-insensitive).  When *repo* is
    ``None`` and a cross-repo ref is found, it is skipped — the caller must
    supply *repo* to validate cross-repo refs.
    """
    target = str(issue_number)
    for m in _CLOSING_KEYWORD_RE.finditer(text):
        ref_repo = m.group(1)  # None for bare #N refs; "owner/repo" otherwise
        ref_num = m.group(2)
        if ref_num != target:
            continue
        if ref_repo is None:
            return True  # bare #N — matches regardless of repo context
        # Cross-repo ref: only match when the caller supplied a repo to compare
        if repo is not None and ref_repo.lower() == repo.lower():
            return True
    return False


def _pr_state_str(pr: dict[str, object]) -> str:
    """Return the GitHub PR state string ("OPEN", "CLOSED", or "MERGED").

    Prefers the explicit ``state`` field from the GraphQL response.  Falls
    back to deriving the state from the ``merged`` boolean when ``state`` is
    absent or ``None`` (defensive for older query shapes).
    """
    state = pr.get("state")
    if state:
        return str(state)
    return "MERGED" if pr.get("merged") else "CLOSED"


# Retry schedule for transient GitHub failures on idempotent GETs (#664).
# Delays in seconds between successive attempts.  Total retry budget is
# ~14s of wall clock on top of the per-request _HTTP_TIMEOUT.  Only applied
# to read-only paths (GET); mutations stay fail-fast.
_GET_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0, 10.0)
_RETRYABLE_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})

# Only requests under ``/repos/{owner}/{repo}/...`` may enter the
# transport-level GET cache.  The per-repo wipe (called on PR
# transitions) is the cache's only lifecycle hook, so caching
# endpoints that aren't owned by a repo — ``/user``, ``/users/{login}``,
# ``/search/issues``, ``/rate_limit``, etc. — would grow unbounded
# without a parallel LRU mechanism.  Scoping the cache to the URLs
# the wipe knows about keeps "no LRU needed" honest.
_CACHEABLE_URL_RE: re.Pattern[str] = re.compile(
    r"^https://api\.github\.com/repos/(?P<repo>[^/]+/[^/]+)/"
)


class _TimeoutSession(_requests.Session):
    """``requests.Session`` with a uniform 30s timeout AND a per-repo
    ETag-validating GET cache.

    Cache semantics
    ===============

    Every ``GET`` against ``api.github.com/repos/{owner}/{name}/...``
    stores its body keyed by the request URL, alongside the response's
    ``ETag`` and ``Last-Modified`` headers (if present).  The next GET
    of the same URL replays those headers as ``If-None-Match`` /
    ``If-Modified-Since``; GitHub returns ``304 Not Modified`` when
    nothing changed, and we return the cached body.  ``304`` responses
    do not count against the primary rate limit, so the cache is a
    free correctness win on every read hot path — the server stays
    authoritative (no webhook-staleness reasoning) and the client pays
    one round-trip rather than a full body fetch.

    Per-repo scope + lifecycle
    --------------------------

    Entries are partitioned by the ``owner/name`` segment of the URL.
    :meth:`clear_repo_cache` drops all entries for one repo, called by
    the worker on PR transitions (open / close / handoff) and by the
    webhook handler on ``pull_request.closed`` — so within a PR's
    lifetime the cache size is bounded by "URLs touched while working
    on this PR", which is small and self-limiting.  No LRU eviction
    needed because the PR-transition wipe is the natural bound.

    Thread safety
    -------------

    All cache mutations go through ``_lock`` because the worker thread,
    the webhook handler thread, and any background reconcile thread
    can hit the same session concurrently (Python 3.14t free-threaded
    — no GIL).
    """

    def __init__(self) -> None:
        super().__init__()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def request(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        method: str | bytes,
        url: str | bytes,
        *args: Any,  # noqa: ANN401  # forwarded verbatim to base.request
        **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to base.request
    ) -> _requests.Response:
        kwargs.setdefault("timeout", _HTTP_TIMEOUT)
        method_s = method.decode() if isinstance(method, bytes) else method
        url_s = url.decode() if isinstance(url, bytes) else url
        if method_s.upper() != "GET":
            return self._raw_request(method, url, *args, **kwargs)
        return self._cached_get(url_s, args, kwargs)

    def _raw_request(  # pragma: no cover — real network; tests override via _FakeTimeoutSession
        self,
        method: str | bytes,
        url: str | bytes,
        *args: Any,  # noqa: ANN401  # forwarded verbatim to base.request
        **kwargs: Any,  # noqa: ANN401  # forwarded verbatim to base.request
    ) -> _requests.Response:
        """Single choke-point to :meth:`requests.Session.request`.

        All network calls within :class:`_TimeoutSession` route through here
        so tests can override this one method (via subclassing) to intercept
        every outbound request without patching the ``requests`` module.
        """
        return super().request(method, url, *args, **kwargs)

    def _cached_get(
        self,
        url: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> _requests.Response:
        cacheable = _CACHEABLE_URL_RE.match(url) is not None
        if not cacheable:
            return self._raw_request("GET", url, *args, **kwargs)
        with self._lock:
            cached = self._cache.get(url)
        headers = dict(kwargs.get("headers") or {})
        if cached is not None:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            if cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified
            kwargs["headers"] = headers
        resp = self._raw_request("GET", url, *args, **kwargs)
        if resp.status_code == 304 and cached is not None:
            return cached.replay(resp)
        if resp.status_code == 200 and (
            resp.headers.get("ETag") or resp.headers.get("Last-Modified")
        ):
            with self._lock:
                self._cache[url] = _CacheEntry.from_response(resp)
                self._cache.move_to_end(url)
        return resp

    def clear_repo_cache(self, repo: str) -> int:
        """Drop every cached entry whose URL belongs to ``repo``.

        ``repo`` is ``owner/name``.  Returns the number of entries
        dropped, for logging / metrics.  Called on PR transitions
        (worker switching to a different issue / PR, or webhook
        observing ``pull_request.closed``) to keep the cache bounded
        without LRU machinery.

        Cache-eligible URLs are restricted to ``/repos/{owner}/{repo}/...``
        (see ``_CACHEABLE_URL_RE``), so this wipe is complete — there
        are no orphan entries from ``/user`` / ``/search/issues`` /
        ``/rate_limit`` etc. that this method couldn't reach.
        """
        prefix = f"/repos/{repo}/"
        with self._lock:
            doomed = [u for u in self._cache if prefix in u]
            for u in doomed:
                del self._cache[u]
        return len(doomed)

    def clear_all(self) -> int:
        """Drop every cached entry across all repos.

        Called by :meth:`GitHub.refresh_token` when the gh-CLI token
        actually rotates — without this, the next GET would revalidate
        with an ``If-None-Match`` validated under the *old* token and
        a server-side 304 would replay the old body, weakening the
        ``#1207`` identity guard that ``Worker.assert_git_identity``
        relies on (it calls ``refresh_token`` then immediately reads
        the authenticated identity).
        """
        with self._lock:
            n = len(self._cache)
            self._cache.clear()
        return n


class _CacheEntry:
    """One cached ``GET`` response — body + revalidation headers."""

    __slots__ = ("etag", "last_modified", "content", "headers")

    def __init__(
        self,
        etag: str,
        last_modified: str,
        content: bytes,
        headers: Mapping[str, str],
    ) -> None:
        self.etag = etag
        self.last_modified = last_modified
        self.content = content
        self.headers = dict(headers)

    @classmethod
    def from_response(cls, resp: _requests.Response) -> "_CacheEntry":
        return cls(
            etag=resp.headers.get("ETag", ""),
            last_modified=resp.headers.get("Last-Modified", ""),
            content=resp.content,
            headers=dict(resp.headers),
        )

    def replay(self, not_modified: _requests.Response) -> _requests.Response:
        """Materialize a fresh 200 ``Response`` carrying the cached body.

        ``requests`` returns the live 304 response to the caller by
        default, which has no body — every GitHub helper would have to
        learn about 304.  Instead we synthesize a 200 with the cached
        body so the rest of the codebase reads ``resp.json()`` /
        ``resp.content`` uniformly.
        """
        replayed = _requests.Response()
        replayed.status_code = 200
        replayed._content = self.content  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        replayed.headers.update(self.headers)
        replayed.url = not_modified.url
        replayed.request = not_modified.request
        return replayed


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
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environ: Mapping[str, str] = os.environ,
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
        token_fetcher: Callable[[], str] = _gh_token,
    ) -> None:
        self._s = session if session is not None else _TimeoutSession()
        self._token_fetcher = token_fetcher
        self._token = token if token is not None else token_fetcher()
        self._s.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        self._sleep = sleeper

    def clear_repo_cache(self, repo: str) -> int:
        """Drop all cached GET responses for ``repo`` from the session.

        Called by the worker on PR transitions (open / close / handoff)
        and by the webhook handler on ``pull_request.closed`` to bound
        the in-process ETag cache without LRU machinery — within a
        PR's lifetime the cache holds only URLs touched while working
        on that PR.

        Returns the number of entries dropped, or ``0`` if the session
        does not expose a cache (e.g. an injected test session).
        """
        if isinstance(self._s, _TimeoutSession):
            return self._s.clear_repo_cache(repo)
        return 0

    def refresh_token(self) -> bool:
        """Re-resolve the gh-CLI token; update session headers if changed.

        Catches the case where the host runs ``gh auth switch`` while
        fido is running: without this, the GitHub client keeps using
        the old token forever, the API's ``/user`` response stays
        wrong, and :meth:`Worker.assert_git_identity` crash-loops the
        worker until a process restart (closes #1207).

        Called at every assertion boundary in the worker loop; the
        per-call cost is one ``gh auth token`` subprocess invocation,
        negligible at the natural iteration cadence.

        Returns ``True`` when the token actually changed.
        """
        new_token = self._token_fetcher()
        if new_token == self._token:
            return False
        self._token = new_token
        self._s.headers["Authorization"] = f"Bearer {new_token}"
        # Wipe every cached response so the next GET re-authenticates
        # with the new token from scratch.  Without this the URL-keyed
        # ETag cache would replay bodies validated under the *old*
        # token on a 304 — bypassing the #1207 identity guard that
        # ``Worker.assert_git_identity`` relies on (it calls
        # ``refresh_token`` then ``get_authenticated_identity``).
        if isinstance(self._s, _TimeoutSession):
            self._s.clear_all()
        return True

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

    def _get(self, path: str) -> Any:  # noqa: ANN401  # JSON dict
        return self._retryable_get(f"{self.BASE}{path}").json()

    def _post(self, path: str, **payload: Any) -> None:  # noqa: ANN401  # JSON pass-through
        resp = self._s.post(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()

    def _post_json(self, path: str, **payload: Any) -> Any:  # noqa: ANN401  # JSON dict
        resp = self._s.post(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _patch(self, path: str, **payload: Any) -> Any:  # noqa: ANN401  # JSON dict
        resp = self._s.patch(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, **payload: Any) -> Any:  # noqa: ANN401  # JSON dict
        resp = self._s.put(f"{self.BASE}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _graphql(self, query: str, **variables: Any) -> Any:  # noqa: ANN401  # JSON dict
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
        **variables: Any,  # noqa: ANN401  # GraphQL JSON pass-through
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

    def list_reactions(
        self, repo: str, comment_type: str, comment_id: int | str
    ) -> list[dict[str, Any]]:
        """Return all reactions on a comment.

        Each item is a dict with at least ``id`` (int) and ``content`` (str).
        comment_type is ``'pulls'`` for review comments or ``'issues'`` for
        top-level PR/issue comments.
        """
        return list(
            self._paginate(
                f"{self.BASE}/repos/{repo}/{comment_type}/comments/{comment_id}/reactions"
            )
        )

    def delete_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        reaction_id: int | str,
    ) -> None:
        """Delete a reaction from a comment by its reaction id.

        comment_type is ``'pulls'`` for review comments or ``'issues'`` for
        top-level PR/issue comments.
        """
        resp = self._s.delete(
            f"{self.BASE}/repos/{repo}/{comment_type}/comments/{comment_id}/reactions/{reaction_id}"
        )
        resp.raise_for_status()

    def reply_to_review_comment(
        self, repo: str, pr: int | str, body: str, in_reply_to: int | str
    ) -> dict[str, Any]:
        """Post a reply to an inline review comment."""
        return self._post_json(
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

    def get_pull_reviews(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return every review submission on a PR as raw API objects.

        Distinct from :meth:`get_reviews` which projects into a narrow
        ``{state, author, submittedAt, ...}`` summary — callers that
        need the full review object (body, html_url, ...) for the
        CommentCache hydration path (#1756) want the raw shape.
        """
        return list(self._paginate(f"{self.BASE}/repos/{repo}/pulls/{pr}/reviews"))

    def fetch_sibling_threads(self, repo: str, pr: int | str) -> list[dict[str, Any]]:
        """Return all review-comment threads for a PR as a structured list.

        Each entry is {path, line, comments: [{author, body}, ...]}.
        Root comments (no in_reply_to_id) start a new thread; replies are appended.
        Raises on network or HTTP errors — callers must not treat failure as empty.
        """
        raw = self.get_pull_comments(repo, pr)

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

        Returns [{author, body}, ...] in posting order.
        Raises on network or HTTP errors — callers must not treat failure as empty.
        """
        raw = self.get_pull_comments(repo, pr)

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
                "in_reply_to_id": c.get("in_reply_to_id"),
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
                    if not (
                        _has_closing_keyword(body, issue_number)
                        or _has_closing_keyword(title, issue_number)
                    ):
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

    def find_closed_unmerged_prs_for_issue(
        self, repo: str, issue_number: int | str, user: str
    ) -> list[int]:
        """Return PR numbers of closed-not-merged PRs this *user* authored
        for *issue_number* (oldest first).

        Used by the fresh-retry path: when an issue has a prior closed-not-
        merged PR, fido must start over from scratch — new branch, new
        triage, new task list — acknowledging the rejection in the pickup
        comment.  See FidoCanCode/home#802.

        Applies the same timeline scan as :meth:`find_pr`: only PRs that
        reference the issue via a closing keyword in their body or via the
        sidebar ``ConnectedEvent`` (with matching ``DisconnectedEvent``
        removal) count.  Filters by author = *user* so an unrelated human
        PR on the same branch never triggers a retry comment.
        """
        owner, name = repo.split("/", 1)
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
            " ...on PullRequest{number state merged title body author{login}}}}"
            "...on ConnectedEvent{subject{__typename"
            " ...on PullRequest{number state merged author{login}}}}"
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
                return []
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
                    if not (
                        _has_closing_keyword(body, issue_number)
                        or _has_closing_keyword(title, issue_number)
                    ):
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
        closed_unmerged: list[int] = []
        for pr_num in sorted(eligible):
            pr = pr_cache[pr_num]
            if pr.get("state") == "CLOSED" and not pr.get("merged"):
                closed_unmerged.append(pr_num)
        return closed_unmerged

    def find_closed_prs_as_context(
        self, repo: str, issue_number: int | str, user: str
    ) -> list[ClosedPR]:
        """Return :class:`~fido.types.ClosedPR` snapshots for closed-not-merged
        PRs linked to *issue_number* by *user*, oldest first.

        Delegates the timeline scan to
        :meth:`find_closed_unmerged_prs_for_issue` to get PR numbers, then
        fetches title and body for each via the REST API.  Used to populate
        the ``prior_attempts`` block in the active-work context so the agent
        can learn from earlier failed attempts.
        """
        pr_numbers = self.find_closed_unmerged_prs_for_issue(repo, issue_number, user)
        result: list[ClosedPR] = []
        for pr_num in pr_numbers:
            data = self._get(f"/repos/{repo}/pulls/{pr_num}")
            result.append(
                ClosedPR(
                    number=int(pr_num),
                    title=data.get("title") or "",
                    body=data.get("body") or "",
                    close_reason="",
                )
            )
        return result

    def _find_linked_pr_for_issue(
        self, repo: str, number: int | str
    ) -> tuple[int | None, bool, str | None]:
        """Scan the timeline of *number* in *repo* for any linked PR.

        Unlike :meth:`find_pr` and :meth:`find_closed_unmerged_prs_for_issue`,
        this helper does **not** filter by author — sub-issue PRs may have been
        authored by anyone (a human, a bot, or a different fido user).

        Returns ``(pr_number, merged, pr_repo)`` for the best linked PR found,
        where *merged* is ``True`` when the PR was merged and *pr_repo* is the
        ``"owner/name"`` repository the PR lives in (which may differ from
        *repo* for cross-repo sub-issues).  Returns ``(None, False, None)``
        when no PR is linked.

        Priority is tiered: merged PRs beat closed-unmerged ones regardless of
        keyword vs sidebar origin.  Within the same merge tier, keyword PRs
        (``CrossReferencedEvent`` with a closing keyword) take priority over
        sidebar PRs (``ConnectedEvent``).  Ties broken by lowest PR number.
        ``DisconnectedEvent`` removes sidebar links.
        """
        owner, name = repo.split("/", 1)
        query = """\
query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    issue(number: $number) {
      timelineItems(
        first: 100
        itemTypes: [
          CROSS_REFERENCED_EVENT
          CONNECTED_EVENT
          DISCONNECTED_EVENT
        ]
        after: $cursor
      ) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          __typename
          ... on CrossReferencedEvent {
            willCloseTarget
            source {
              __typename
              ... on PullRequest {
                number
                state
                merged
                body
                title
                repository {
                  nameWithOwner
                }
              }
            }
          }
          ... on ConnectedEvent {
            subject {
              __typename
              ... on PullRequest {
                number
                state
                merged
                repository {
                  nameWithOwner
                }
              }
            }
          }
          ... on DisconnectedEvent {
            subject {
              __typename
              ... on PullRequest {
                number
                repository {
                  nameWithOwner
                }
              }
            }
          }
        }
      }
    }
  }
}
"""
        # GitHub PR numbers are only unique inside a repository.  Key linked
        # candidates by repo and number so cross-repo sub-issue PRs cannot
        # collide while we choose or disconnect candidates.  Values are the
        # GitHub PR state string: "OPEN", "CLOSED", or "MERGED".
        keyword_prs: dict[tuple[str, int], str] = {}
        sidebar_prs: dict[tuple[str, int], str] = {}
        cursor: str | None = None
        while True:
            data = self._graphql(
                query,
                owner=owner,
                repo=name,
                number=int(number),
                cursor=cursor,
            )
            items = data["data"]["repository"]["issue"]["timelineItems"]
            if not items:
                break
            for node in items["nodes"]:
                typename = node["__typename"]
                if typename == "CrossReferencedEvent":
                    pr = node.get("source") or {}
                    if pr.get("__typename") != "PullRequest":
                        continue
                    # Treat as a keyword/closing PR when willCloseTarget is
                    # true, OR when the PR body/title contains a closing
                    # keyword referencing this issue.  willCloseTarget is
                    # false once the target issue is already closed, so a
                    # pure willCloseTarget check incorrectly rejects real
                    # closing PRs for already-closed sub-issues.  The
                    # keyword fallback catches those; bare mentions ("see
                    # #42") that lack a keyword are still skipped.
                    will_close = node.get("willCloseTarget", False)
                    pr_body = pr.get("body") or ""
                    pr_title = pr.get("title") or ""
                    if not will_close and not _has_closing_keyword(
                        pr_body + " " + pr_title, number, repo
                    ):
                        continue
                    pr_num = pr["number"]
                    pr_repo_val = (pr.get("repository") or {}).get(
                        "nameWithOwner"
                    ) or repo
                    pr_key = (pr_repo_val, pr_num)
                    if pr_key not in keyword_prs:
                        keyword_prs[pr_key] = _pr_state_str(pr)
                elif typename == "ConnectedEvent":
                    pr = node.get("subject") or {}
                    if pr.get("__typename") != "PullRequest":
                        continue
                    pr_num = pr["number"]
                    pr_repo_val = (pr.get("repository") or {}).get(
                        "nameWithOwner"
                    ) or repo
                    pr_key = (pr_repo_val, pr_num)
                    if pr_key not in sidebar_prs:
                        sidebar_prs[pr_key] = _pr_state_str(pr)
                elif typename == "DisconnectedEvent":
                    pr = node.get("subject") or {}
                    if pr.get("__typename") == "PullRequest":
                        pr_repo_val = (pr.get("repository") or {}).get(
                            "nameWithOwner"
                        ) or repo
                        sidebar_prs.pop((pr_repo_val, pr["number"]), None)
            page_info = items["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            cursor = page_info["endCursor"]
        # Filter OPEN PRs from each bucket first — a sub-issue manually
        # closed while its PR is still open has no completed PR body worth
        # including.
        kw_candidates = {k: s for k, s in keyword_prs.items() if s != "OPEN"}
        sb_candidates = {k: s for k, s in sidebar_prs.items() if s != "OPEN"}
        # Tier by merge state across both buckets: merged beats
        # closed-unmerged regardless of keyword vs sidebar origin.  Within
        # the same merge tier, keyword PRs take priority over sidebar PRs.
        kw_merged = {k: s for k, s in kw_candidates.items() if s == "MERGED"}
        sb_merged = {k: s for k, s in sb_candidates.items() if s == "MERGED"}
        chosen = kw_merged or sb_merged or kw_candidates or sb_candidates
        if not chosen:
            return None, False, None
        # Break ties by lowest number, then repository.
        pr_repo, pr_num = min(chosen, key=lambda pr_key: (pr_key[1], pr_key[0]))
        pr_merged = chosen[(pr_repo, pr_num)] == "MERGED"
        return pr_num, pr_merged, pr_repo

    def fetch_closed_sub_issues(
        self, repo: str, number: int | str
    ) -> list[ClosedSubIssue]:
        """Return :class:`~fido.types.ClosedSubIssue` entries for each closed
        direct sub-issue of *number* in *repo*, oldest-first.

        Calls ``GET /repos/{repo}/issues/{number}/sub_issues`` and filters to
        items whose ``state`` is ``"closed"``.  For each, walks the timeline
        via :meth:`_find_linked_pr_for_issue` to discover any linked PR (any
        author — sub-issue PRs are not filtered by user).  Fetches the PR body
        via the REST API when a PR is found.

        No recursion — direct children only.  Every parent issue summarises
        its children; traversing further would be prohibitively expensive.
        """
        items = list(
            self._paginate(f"{self.BASE}/repos/{repo}/issues/{number}/sub_issues")
        )
        result: list[ClosedSubIssue] = []
        for item in items:
            if item.get("state") != "closed":
                continue
            sub_num = int(item["number"])
            sub_title = item.get("title") or ""
            sub_body = item.get("body") or ""
            state_reason: str | None = item.get("state_reason") or None
            pr_num, pr_merged, pr_repo = self._find_linked_pr_for_issue(repo, sub_num)
            if pr_num is None:
                close_state = "closed_no_pr"
                pr_body = ""
            else:
                close_state = "merged" if pr_merged else "closed_unmerged"
                pr_data = self._get(f"/repos/{pr_repo}/pulls/{pr_num}")
                pr_body = pr_data.get("body") or ""
            result.append(
                ClosedSubIssue(
                    number=sub_num,
                    title=sub_title,
                    body=sub_body,
                    close_state=close_state,
                    state_reason=state_reason,
                    pr_number=pr_num,
                    pr_repo=pr_repo,
                    pr_body=pr_body,
                )
            )
        return result

    def comment_issue(self, repo: str, number: int | str, body: str) -> dict[str, Any]:
        """Post a comment on an issue."""
        return self._post_json(f"/repos/{repo}/issues/{number}/comments", body=body)

    def close_issue(self, repo: str, number: int | str) -> None:
        """Close an issue by setting state=closed."""
        self._patch(f"/repos/{repo}/issues/{number}", state="closed")

    def close_pr(self, repo: str, pr: int | str) -> None:
        """Close a PR (without merging) by setting state=closed."""
        self._patch(f"/repos/{repo}/pulls/{pr}", state="closed")

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

    def get_authenticated_identity(self) -> GitIdentity:
        """Return the git commit identity derived from the authenticated GitHub user.

        Name: the account's display name, falling back to ``login`` when unset.
        Email: the GitHub noreply form
        ``{id}+{login}@users.noreply.github.com`` — never the real email.
        """
        data = self._get("/user")
        login = data["login"]
        uid = data["id"]
        name = data.get("name") or login
        return GitIdentity(
            name=name,
            email=f"{uid}+{login}@users.noreply.github.com",
        )

    def get_user_identity(self, login: str) -> GitIdentity:
        """Return the git commit identity for an arbitrary GitHub user by login.

        Name: the account's display name, falling back to ``login`` when unset.
        Email: the GitHub noreply form
        ``{id}+{login}@users.noreply.github.com`` — never the real email.
        """
        data = self._get(f"/users/{login}")
        uid = data["id"]
        name = data.get("name") or login
        return GitIdentity(
            name=name,
            email=f"{uid}+{login}@users.noreply.github.com",
        )

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
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
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
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
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
        return list(
            self._graphql_paginate(
                query,
                ("repository", "issues"),
                owner=owner,
                repo=repo,
            )
        )

    def add_assignee(self, repo: str, number: int | str, login: str) -> None:
        """Assign *login* to issue *number* in *repo*.

        Uses the REST endpoint so assignment is additive — existing
        assignees are preserved.
        """
        self._post(f"/repos/{repo}/issues/{number}/assignees", assignees=[login])

    def get_rate_limit(self) -> dict[str, Any]:
        """Return GitHub's per-resource rate-limit snapshot.

        Hits ``GET /rate_limit`` — per GitHub docs this endpoint does not
        itself count against any quota, so it's safe to poll on a 60s
        cadence (closes #812 follow-up).  Returns the raw ``resources``
        dict; callers parse the windows they care about (``core``,
        ``graphql``, etc.).
        """
        data = self._get("/rate_limit")
        return data["resources"]

    def view_issue(self, repo: str, number: int | str) -> dict[str, Any]:
        """Return issue data (state, title, body, created_at, labels)."""
        data = self._get(f"/repos/{repo}/issues/{number}")
        return {
            "state": data["state"].upper(),
            "title": data["title"],
            "body": data["body"] or "",
            "created_at": data.get("created_at", ""),
            "labels": [lbl["name"] for lbl in data.get("labels", [])],
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

    def create_issue(
        self,
        repo: str,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> str:
        """Create an issue and return its HTML URL."""
        extra: dict[str, Any] = {}
        if labels:
            extra["labels"] = labels
        data = self._post_json(f"/repos/{repo}/issues", title=title, body=body, **extra)
        return data["html_url"]

    def search_issues(self, repo: str, query: str) -> list[dict[str, Any]]:
        """Search issues in *repo* matching *query* and return the result items.

        Prepends ``repo:{repo}`` to *query* so callers do not need to repeat
        the repo qualifier.  Returns the ``items`` list from the GitHub search
        response — an empty list when nothing matches.
        """
        q = f"repo:{repo} {query}"
        url = f"{self.BASE}/search/issues?{urllib.parse.urlencode({'q': q})}"
        data = self._retryable_get(url).json()
        return list(data.get("items", []))

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

    def get_pr_state(self, repo: str, pr: int | str) -> str:
        """Return just the PR state (``"open"`` or ``"closed"``).

        One ``_get`` call, no review/commit pagination — used by the orphan
        comment-queue sweep (#1691) which only needs to know whether the PR
        is still open.
        """
        data = self._get(f"/repos/{repo}/pulls/{pr}")
        return data["state"]

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
        raising.  This handles the race where fido self-restarts after a
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

    def try_enable_auto_merge(
        self, repo: str, pr: int | str, squash: bool = True
    ) -> bool:
        """Enable GitHub's native auto-merge on *pr* without falling back to
        an immediate REST merge (fix for #787).

        Returns ``True`` if auto-merge was enabled, ``False`` when the repo
        has auto-merge disabled (GraphQL ``UNPROCESSABLE``).  Unlike
        :meth:`pr_merge` with ``auto=True``, this does not attempt the REST
        merge on failure — callers using this in the "mark ready, not yet
        approved" path must not trigger the REST path, which would 405 on
        the pending review requirement.
        """
        pr_data = self._get(f"/repos/{repo}/pulls/{pr}")
        if pr_data.get("merged"):
            return False
        return self._try_enable_auto_merge(repo, pr, pr_data, squash)

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
        """Return PR data (title, reviews, isDraft, mergeStateStatus, body, commits)."""
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
            "title": pr_data["title"],
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
        """Return review-thread nodes for a PR.

        Bot authors (``__typename == "Bot"``) have their ``login``
        normalized to append the ``[bot]`` suffix.  GitHub's GraphQL
        strips ``[bot]`` from bot logins while REST and webhook payloads
        include it; this normalization keeps downstream consumers
        (``_filter_threads``, allowlists, oracles) on a single
        identifier shape regardless of which API supplied the data.
        Closes #1624.
        """
        query = (
            "query($owner:String!,$repo:String!,$pr:Int!){"
            "repository(owner:$owner,name:$repo){"
            "pullRequest(number:$pr){"
            "reviewThreads(first:100){"
            "nodes{id isResolved"
            " comments(first:50){nodes{author{__typename login} body url createdAt databaseId}}"
            "}}}}}"
        )
        data = self._graphql(query, owner=owner, repo=repo, pr=int(pr))
        nodes = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
        for node in nodes:
            for comment in node.get("comments", {}).get("nodes", []):
                author = comment.get("author")
                if author is None:
                    continue
                if author.get("__typename") != "Bot":
                    continue
                login = author.get("login", "")
                if login and not login.endswith("[bot]"):
                    author["login"] = f"{login}[bot]"
        return nodes

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
