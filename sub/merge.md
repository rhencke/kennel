The branch has a merge conflict with upstream. All context (PR, repo, branch, upstream) is in the Context section above.

## Steps

### 1. Fetch and merge upstream
```bash
git fetch origin
git merge origin/<default_branch>
```

If the merge succeeds cleanly (exit 0, no conflicts), skip to step 3.

### 2. Resolve conflicts

Find all conflicted files:
```bash
git status --short | grep '^UU\|^AA\|^DD\|^AU\|^UA\|^DU\|^UD' | awk '{print $2}'
```

For each conflicted file:
1. Read the file. Understand both sides of the conflict (ours vs theirs).
2. Resolve the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) by writing the correct merged result.
3. Do NOT leave any conflict markers in the file.

After resolving all files:
```bash
git add <resolved-files>
```

### 3. Commit and push
```bash
git commit -m "Merge origin/<default_branch> into <branch>"
git push
```

## Done when
Merge complete, no conflict markers remain, committed and pushed.

## Constraints
- **Never** rebase, amend, or force-push. New commits only.
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, TodoRead, or `kennel task`.
- **Never** edit the PR body directly.
