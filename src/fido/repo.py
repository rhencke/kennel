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
    """One managed repository's collaborators."""

    name: str
    work_dir: Path
    tasks: Tasks
    state: State
