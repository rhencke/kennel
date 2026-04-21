from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "build"


def run_build(
    tmp_path: Path, output: Path, *command: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BUILD), "--smart-output", str(output), "--", *command],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )


class TestBuildWrapper:
    def test_removes_stale_files_after_new_output(self, tmp_path: Path) -> None:
        output = tmp_path / "out"
        output.mkdir()
        stale = output / "stale.py"
        stale.write_text("old\n")
        time.sleep(0.01)

        result = run_build(
            tmp_path,
            output,
            "sh",
            "-c",
            'sleep 0.1; printf "new\\n" > "$1/new.py"',
            "sh",
            str(output),
        )

        assert result.returncode == 0, result.stderr
        assert not stale.exists()
        assert (output / "new.py").read_text() == "new\n"

    def test_preserves_output_when_command_fails(self, tmp_path: Path) -> None:
        output = tmp_path / "out"
        output.mkdir()
        existing = output / "existing.py"
        existing.write_text("keep\n")

        result = run_build(tmp_path, output, "sh", "-c", "exit 7")

        assert result.returncode == 7
        assert existing.read_text() == "keep\n"

    def test_fails_without_new_output_and_preserves_files(self, tmp_path: Path) -> None:
        output = tmp_path / "out"
        output.mkdir()
        existing = output / "existing.py"
        existing.write_text("keep\n")

        result = run_build(tmp_path, output, "true")

        assert result.returncode == 1
        assert "no newly produced files" in result.stderr
        assert existing.read_text() == "keep\n"

    def test_passes_command_arguments_through(self, tmp_path: Path) -> None:
        output = tmp_path / "out"
        output.mkdir()

        result = run_build(
            tmp_path,
            output,
            "sh",
            "-c",
            'printf "%s\\n" "$2" > "$1/arg.txt"; printf "arg.txt\\n" > "$1/.build-files"',
            "sh",
            str(output),
            "passed",
        )

        assert result.returncode == 0, result.stderr
        assert (output / "arg.txt").read_text() == "passed\n"

    def test_manifest_preserves_current_files_and_removes_unlisted_files(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        current = output / "current.py"
        stale = output / "stale.py"
        current.write_text("old but current\n")
        stale.write_text("old and stale\n")
        time.sleep(0.01)

        result = run_build(
            tmp_path,
            output,
            "sh",
            "-c",
            'printf "current.py\\nnew.py\\n" > "$1/.build-files"; printf "new\\n" > "$1/new.py"',
            "sh",
            str(output),
        )

        assert result.returncode == 0, result.stderr
        assert current.read_text() == "old but current\n"
        assert (output / "new.py").read_text() == "new\n"
        assert not stale.exists()
        assert not (output / ".build-files").exists()


class TestModelsBuildScript:
    def test_uses_buildx_without_docker_run_or_host_uv(self) -> None:
        script = BUILD.read_text()

        assert "docker buildx build" in script
        assert "docker run" not in script
        assert "docker cp" not in script
        assert "uv run" not in script
        assert "dune build" not in script

    def test_passes_cache_context_and_local_output(self) -> None:
        script = BUILD.read_text()

        assert '--build-context "rocq_models_cache=$cache_context"' in script
        assert "--output type=local,dest=." in script
        assert "--smart-output kennel/models_generated" in script
        assert "--file rocq-python-extraction/Dockerfile" in script
        assert '--cache-to "type=local,dest=$image_cache_next,mode=max"' in script
        assert '--build-arg "ROCQ_IMAGE=$rocq_image"' in script

    def test_scripts_are_executable(self) -> None:
        assert os.access(BUILD, os.X_OK)
