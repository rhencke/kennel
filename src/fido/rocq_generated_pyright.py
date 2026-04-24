"""Run pyright over generated Rocq extraction artifacts."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_dir", type=Path)
    parser.add_argument(
        "--checks-dir",
        type=Path,
        default=Path("rocq-python-extraction/test"),
    )
    args = parser.parse_args()

    generated_dir = args.generated_dir
    checks = sorted(args.checks_dir.glob("pyright_*_check.py"))
    for check in checks:
        shutil.copy2(check, generated_dir / check.name)

    config = {
        "include": [check.name for check in checks],
        "executionEnvironments": [{"root": ".", "extraPaths": ["."]}],
        "reportUnusedImport": False,
        "reportUnusedVariable": False,
        "reportUnknownLambdaType": False,
        "reportRedeclaration": False,
    }
    config_path = generated_dir / "pyrightconfig.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    os.execvp("pyright", ["pyright", "-p", str(config_path)])
    raise RuntimeError("pyright exec failed")


if __name__ == "__main__":  # pragma: no cover
    main()
