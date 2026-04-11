#!/usr/bin/env bash
# Generate _data/stats/<from>.yml from FidoCanCode's git activity.
#
# Usage:
#   scripts/generate-stats.sh <date>             # single day
#   scripts/generate-stats.sh <from> <to>        # date range, inclusive
#
# Dates are ISO format (YYYY-MM-DD), interpreted as UTC.  The output file
# is keyed on <from> so each blog post has a stable companion data file.
# Both index and post layouts read this via site.data.stats[date].
#
# Sources:
#  - Commits: `git log --reflog --all --since --until --author` against each
#    managed clone.  --reflog walks every commit that ever existed in the
#    local object database, including ones from squash-merged branches that
#    were force-deleted from origin (the events API misses many of these
#    after retention; the contribution graph counts squash merges as 1 each).
#  - PRs / issues / reviews: GraphQL contributionsCollection totals — those
#    counts are accurate regardless of branch state.
#
# Identity transition: before 2026-04-06 Fido's work was committed under
# "Robert Hencke" / rhencke.  For windows entirely before that date the
# script uses the pre-transition identity so April 5 (Fido's first day) is
# credited correctly.
#
# Requires: gh, git, python3.  Each managed clone must already exist locally
# at the path under MANAGED_PATHS below.
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

repo_root=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
out="$repo_root/_data/stats/$from.yml"
mkdir -p "$(dirname "$out")"

# Map of "owner/name" → local clone path.  Edit when adding repos.
declare -A MANAGED_PATHS=(
  ["rhencke/confusio"]="/home/rhencke/workspace/confusio"
  ["rhencke/kennel"]="/home/rhencke/workspace/kennel"
  ["FidoCanCode/fidocancode.github.io"]="/home/rhencke/workspace/fidocancode.github.io"
)

# Before 2026-04-06, work was committed under Robert Hencke / rhencke before
# Fido's identity was established.  We credit those contributions to Fido by
# using the pre-transition author name and GitHub login for any window that
# falls entirely before the transition date.
fido_login="FidoCanCode"
fido_git_author="Fido Can Code"
pre_transition_login="rhencke"
pre_transition_git_author="Robert Hencke"
transition_date="2026-04-06"

since="${from}T00:00:00Z"
until="${to}T23:59:59Z"

# Pick which git author and GraphQL login to use.  If the entire requested
# window predates the transition, use the pre-transition identity; otherwise
# use Fido's identity.  (Mixed windows — e.g. a weekly retrospective spanning
# the transition — keep Fido's identity; the pre-transition day will simply
# contribute zero from those sources, which is already the committed status.)
if [[ "$to" < "$transition_date" ]]; then
  login="$pre_transition_login"
  git_author="$pre_transition_git_author"
else
  login="$fido_login"
  git_author="$fido_git_author"
fi

total_commits=0
declare -a touched_repos=()

for repo in "${!MANAGED_PATHS[@]}"; do
  path=${MANAGED_PATHS[$repo]}
  if [[ ! -d "$path/.git" ]]; then
    echo "warning: clone missing for $repo at $path — skipping" >&2
    continue
  fi
  # --reflog includes commits from deleted branches still in the object db.
  # --all includes all refs.  Together they catch nearly everything fido
  # produced locally, even after squash merges deleted the source branch.
  count=$(git -C "$path" log \
    --reflog \
    --all \
    --since="$since" \
    --until="$until" \
    --author="$git_author" \
    --pretty=oneline 2>/dev/null \
    | wc -l)
  if [[ "$count" -gt 0 ]]; then
    total_commits=$((total_commits + count))
    touched_repos+=("$repo")
  fi
done

# PRs / issues / reviews via GraphQL contributionsCollection (no pagination
# needed for a window <= 1 year; we'll add looping if/when fido starts
# generating multi-year retrospectives).
totals_query='query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      totalIssueContributions
      totalPullRequestContributions
      totalPullRequestReviewContributions
    }
  }
}'

totals_response=$(gh api graphql \
  -F login="$login" \
  -F from="$since" \
  -F to="$until" \
  -f query="$totals_query")

REPOS_LIST="${touched_repos[*]:-}" \
TOTAL_COMMITS="$total_commits" \
TOTALS_JSON="$totals_response" \
python3 - "$from" "$to" "$out" <<'PY'
import json
import os
import sys

from_date, to_date, out_path = sys.argv[1:4]

commits = int(os.environ["TOTAL_COMMITS"])
repos = sorted(r for r in os.environ["REPOS_LIST"].split() if r)

totals = json.loads(os.environ["TOTALS_JSON"])
collection = totals["data"]["user"]["contributionsCollection"]
issues = collection["totalIssueContributions"]
prs = collection["totalPullRequestContributions"]
reviews = collection["totalPullRequestReviewContributions"]

with open(out_path, "w") as f:
    f.write("# Auto-generated by scripts/generate-stats.sh. Do not edit by hand.\n")
    f.write(f"date: {from_date}\n")
    f.write("date_range:\n")
    f.write(f"  from: {from_date}\n")
    f.write(f"  to: {to_date}\n")
    f.write(f"commits: {commits}\n")
    f.write(f"prs: {prs}\n")
    f.write(f"issues: {issues}\n")
    f.write(f"reviews: {reviews}\n")
    f.write("repos:\n")
    for r in repos:
        f.write(f"  - {r}\n")

print(
    f"wrote {out_path}: {commits} commits, {prs} PRs, "
    f"{issues} issues, {reviews} reviews across {len(repos)} repos"
)
PY
