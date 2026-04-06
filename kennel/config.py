from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("kennel")


@dataclass(frozen=True)
class Config:
    port: int
    secret: bytes
    work_dir: Path
    work_script: Path
    project: str
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        missing = [
            k
            for k in ("KENNEL_SECRET", "KENNEL_WORK_DIR", "KENNEL_PROJECT")
            if not os.environ.get(k)
        ]
        if missing:
            raise SystemExit(f"missing required env vars: {', '.join(missing)}")

        return cls(
            port=int(os.environ.get("KENNEL_PORT", "9000")),
            secret=os.environ["KENNEL_SECRET"].encode(),
            work_dir=Path(os.environ["KENNEL_WORK_DIR"]).expanduser().resolve(),
            work_script=Path(
                os.environ.get(
                    "KENNEL_WORK_SCRIPT",
                    str(Path(__file__).resolve().parent.parent / "work.sh"),
                )
            )
            .expanduser()
            .resolve(),
            project=os.environ["KENNEL_PROJECT"],
            log_level=os.environ.get("KENNEL_LOG_LEVEL", "INFO").upper(),
        )
