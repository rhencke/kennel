"""Per-repo collaborators owned by :class:`~fido.registry.WorkerRegistry`.

One :class:`Repo` instance per managed repository, holding the
collaborators whose lifetime matches the repo's lifetime in this fido
process: the publishing-aware :class:`~fido.tasks.Tasks`,
:class:`~fido.state.State`, and (over time) the other per-repo state
that historically lived as parallel ``dict[str, X]`` fields on the
registry.

The registry exposes the repo via :meth:`~fido.registry.WorkerRegistry.repo_for`
so webhook handlers, the worker, and ``reorder_tasks`` all reach the
same instances — no scattered ``Tasks(work_dir)`` / ``State(fido_dir)``
calls that would silently bypass the on_mutate snapshot publisher.
"""

from dataclasses import dataclass
from pathlib import Path

from fido.state import State
from fido.tasks import Tasks


@dataclass(frozen=True, slots=True)
class Repo:
    """One managed repository's collaborators.

    *git_dir* is the canonical absolute git directory, resolved once via
    ``git rev-parse --absolute-git-dir`` at registry construction.
    Linked worktrees and submodules have ``work_dir/.git`` as a *file*
    pointing to the real git directory elsewhere; resolving once and
    storing the answer here means every consumer (worker, webhook
    handlers, status path) reaches the same directory without
    re-shelling out.
    """

    name: str
    work_dir: Path
    git_dir: Path
    tasks: Tasks
    state: State
