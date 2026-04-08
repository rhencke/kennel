"""Watchdog — check WorkerThread health and restart dead threads."""

from __future__ import annotations

import logging

from kennel.config import RepoConfig
from kennel.registry import WorkerRegistry

log = logging.getLogger(__name__)


class Watchdog:
    """Check whether each WorkerThread is alive and restart any that have died.

    Accepts *registry* and *repos* via the constructor so tests can inject
    mock objects without patching module-level names.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        repos: dict[str, RepoConfig],
    ) -> None:
        self.registry = registry
        self.repos = repos

    def run(self) -> int:
        """Run one watchdog iteration. Returns 0."""
        for repo_name, repo_cfg in self.repos.items():
            if not self.registry.is_alive(repo_name):
                log.info("thread for %s is not alive — restarting", repo_name)
                self.registry.start(repo_cfg)
        return 0


def run(registry: WorkerRegistry, repos: dict[str, RepoConfig]) -> int:
    """Module-level entry point: create a Watchdog and run it."""
    return Watchdog(registry, repos).run()
