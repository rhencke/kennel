#!/usr/bin/env bash
# Generate _data/stats/<from>.yml from FidoCanCode's GitHub activity.
#
# Usage:
#   scripts/generate-stats.sh <date>             # single day
#   scripts/generate-stats.sh <from> <to>        # date range, inclusive
#
# Dates are ISO format (YYYY-MM-DD), interpreted as UTC.  The output file
# is keyed on <from> so each blog post has a stable companion data file.
# Both index and post layouts read this via site.data.stats[date].
#
# Sources — all from GitHub's GraphQL API (no local clones needed):
#  - Commits: PR commit totals (pre-squash) + contribution graph commits.
#    PR totals give the real work done; contrib graph adds direct pushes.
#  - PRs / issues / reviews: contributionsCollection totals
#
# Identity transition: before 2026-04-06 Fido's work was committed under
# "Robert Hencke" / rhencke.  For windows entirely before that date the
# script uses the pre-transition identity so April 5 (Fido's first day) is
# credited correctly.
#
# Requires: gh, python3.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <from> [to]" >&2
  exit 1
fi

from=$1
to=${2:-$1}

for d in "$from" "$to"; do
  if ! [[ $d =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "invalid date: $d (expected YYYY-MM-DD)" >&2
    exit 1
  fi
done

docs_root=$(cd "$(dirname "$0")/.." && pwd -P)
out="$docs_root/_data/stats/$from.yml"
mkdir -p "$(dirname "$out")"

# Repos we track.
MANAGED_REPOS=("rhencke/confusio" "FidoCanCode/home")

# Before 2026-04-06, work was committed under Robert Hencke / rhencke before
# Fido's identity was established.  We credit those contributions to Fido by
# using the pre-transition GitHub login for any window that falls entirely
# before the transition date.
fido_login="FidoCanCode"
pre_transition_login="rhencke"
transition_date="2026-04-06"

since="${from}T00:00:00Z"
until="${to}T23:59:59Z"

if [[ "$to" < "$transition_date" ]]; then
  login="$pre_transition_login"
else
  login="$fido_login"
fi

# Build search string covering all managed repos.
repo_filter=""
for repo in "${MANAGED_REPOS[@]}"; do
  repo_filter+="repo:${repo} "
done

# Hand everything to Python — one process, clean pagination.
MANAGED_REPOS_LIST="${MANAGED_REPOS[*]}" \
python3 - "$login" "$since" "$until" "$from" "$to" "$out" "$repo_filter" <<'PY'
import json
import os
import subprocess
import sys

login, since, until, from_date, to_date, out_path, repo_filter = sys.argv[1:8]
managed = set(os.environ["MANAGED_REPOS_LIST"].split())


def gh_graphql(query, **variables):
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        cmd += ["-F", f"{k}={v}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


# 1. contributionsCollection: commit counts per repo + PR/issue/review totals.
totals = gh_graphql(
    """query($login: String!, $from: DateTime!, $to: DateTime!) {
      user(login: $login) {
        contributionsCollection(from: $from, to: $to) {
          totalIssueContributions
          totalPullRequestContributions
          totalPullRequestReviewContributions
          commitContributionsByRepository {
            repository { nameWithOwner }
            contributions { totalCount }
          }
        }
      }
    }""",
    login=login,
    **{"from": since, "to": until},
)
collection = totals["data"]["user"]["contributionsCollection"]
issues = collection["totalIssueContributions"]
prs = collection["totalPullRequestContributions"]
reviews = collection["totalPullRequestReviewContributions"]

# Baseline commit counts from contribution graph (squash merges = 1 each).
contrib_commits: dict[str, int] = {}
for entry in collection["commitContributionsByRepository"]:
    repo = entry["repository"]["nameWithOwner"]
    if repo in managed:
        contrib_commits[repo] = entry["contributions"]["totalCount"]

# 2. Merged PRs in window — paginate to get commits.totalCount from each.
pr_query = """query($search: String!, $cursor: String) {
  search(query: $search, type: ISSUE, first: 100, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        repository { nameWithOwner }
        commits { totalCount }
      }
    }
  }
}"""

search_str = f"is:pr author:{login} is:merged merged:{from_date}..{to_date} {repo_filter}"
all_nodes = []
cursor = None
while True:
    variables = {"search": search_str}
    if cursor:
        variables["cursor"] = cursor
    page = gh_graphql(pr_query, **variables)
    search_data = page["data"]["search"]
    all_nodes.extend(search_data["nodes"])
    if search_data["pageInfo"]["hasNextPage"]:
        cursor = search_data["pageInfo"]["endCursor"]
    else:
        break

# 3. PR commit totals (pre-squash) + contrib graph commits.
pr_commits_total: dict[str, int] = {}
for node in all_nodes:
    if not node:
        continue
    repo = node["repository"]["nameWithOwner"]
    if repo in managed:
        pr_commits_total[repo] = pr_commits_total.get(repo, 0) + node["commits"]["totalCount"]

total_commits = 0
touched_repos = []
for repo in sorted(managed):
    count = pr_commits_total.get(repo, 0) + contrib_commits.get(repo, 0)
    if count > 0:
        total_commits += count
        touched_repos.append(repo)

# 4. Write output.
with open(out_path, "w") as f:
    f.write("# Auto-generated by scripts/generate-stats.sh. Do not edit by hand.\n")
    f.write(f"date: {from_date}\n")
    f.write("date_range:\n")
    f.write(f"  from: {from_date}\n")
    f.write(f"  to: {to_date}\n")
    f.write(f"commits: {total_commits}\n")
    f.write(f"prs: {prs}\n")
    f.write(f"issues: {issues}\n")
    f.write(f"reviews: {reviews}\n")
    f.write("repos:\n")
    for r in touched_repos:
        f.write(f"  - {r}\n")

print(
    f"wrote {out_path}: {total_commits} commits, {prs} PRs, "
    f"{issues} issues, {reviews} reviews across {len(touched_repos)} repos"
)
PY
