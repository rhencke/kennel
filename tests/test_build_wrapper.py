from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "build"
FIDO = REPO / "fido"
BAKE = REPO / "docker-bake.hcl"
PRE_COMMIT = REPO / ".githooks" / "pre-commit"


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

        assert "buildx_driver()" in script
        assert "warning: docker buildx inspect failed" in script
        assert '--build-context "rocq_models_cache=$cache_context"' in script
        assert '--build-context "rocq_image=oci-layout://$image_oci"' in script
        assert '--output "type=oci,dest=$image_oci_tar"' in script
        assert "--output type=local,dest=." in script
        assert "--smart-output src/fido/rocq" in script
        assert "--file rocq-python-extraction/Dockerfile" in script
        assert '--cache-to "type=local,dest=$image_cache_next,mode=max"' in script
        assert '--cache-to "type=local,dest=$build_cache_next,mode=max"' in script
        assert '--build-arg "ROCQ_IMAGE=$rocq_image"' in script

    def test_explicit_check_targets_skip_local_output_export(self) -> None:
        script = BUILD.read_text()

        assert "requested_target=" in script
        assert '[ "$requested_target" != "export" ]' in script
        ci_branch = script.split(
            'if [ -n "$requested_target" ] && [ "$requested_target" != "export" ]; then',
            maxsplit=1,
        )[1].split("return", maxsplit=1)[0]

        assert "--output type=local,dest=." not in ci_branch

    def test_scripts_are_executable(self) -> None:
        assert os.access(BUILD, os.X_OK)
        assert os.access(FIDO, os.X_OK)

    def test_pre_commit_runs_ci_and_runtime_smoke(self) -> None:
        script = PRE_COMMIT.read_text()

        assert "./fido warm" in script


class TestFidoLauncher:
    def test_maps_friendly_commands_to_project_scripts(self) -> None:
        script = FIDO.read_text()

        assert "help)" in script
        assert "run_container fido-help" in script
        assert "up)" in script
        assert "supervise_up fido --secret-file /run/secrets/fido-secret" in script
        assert "warm)" in script
        assert "warm_images" in script
        assert "status)" in script
        assert "run_container fido-status" in script
        assert "task)" in script
        assert "run_container fido-task" in script
        assert "sync-tasks)" in script
        assert "run_container fido-sync-tasks" in script

    def test_supervises_foreground_container_and_down_stops_by_name(self) -> None:
        script = FIDO.read_text()
        removed_flag = "--detach"

        assert removed_flag not in script
        assert "supervise_up()" in script
        assert "--rm" in script
        assert "fido_log=${FIDO_LOG:-$HOME/log/fido.log}" in script
        assert "redirect_up_logs()" in script
        assert 'exec >>"$fido_log" 2>&1' in script
        assert "named_run=0" in script
        assert "named_run=1" in script
        assert 'if [ "$named_run" = "1" ]; then' in script
        assert 'docker rm -f "$container"' in script
        assert 'docker stop "$container"' in script
        assert 'run_args=(--name "$container" "${run_args[@]}")' in script
        assert 'restart_codes=" 3 75 "' in script
        assert 'stop_codes=" 0 130 137 143 "' in script
        assert "container exited code=$status action=restart" in script
        assert "container exited code=$status action=stop" in script
        assert "container exited code=$status action=fail" in script
        assert "${FIDO_CONTAINER:-fido}" in script

    def test_auto_update_cleans_syncs_and_can_be_disabled(self) -> None:
        script = FIDO.read_text()

        assert "auto_update=${FIDO_AUTO_UPDATE:-1}" in script
        assert "--no-auto-update)" in script
        assert "auto-update disabled; using dirty runner clone as-is" in script
        assert 'git -C "$repo_root" reset --hard' in script
        assert 'git -C "$repo_root" clean -fd -e .cache/' in script
        assert 'git -C "$repo_root" fetch origin main' in script
        assert 'git -C "$repo_root" reset --hard origin/main' in script

    def test_builds_runtime_and_dev_images_with_host_identity(self) -> None:
        script = FIDO.read_text()

        assert "docker buildx bake" in script
        assert "dev_image=${FIDO_DEV_IMAGE:-fido-dev:local}" in script
        assert "builder=${FIDO_BUILDX_BUILDER:-fido}" in script
        assert "ensure_buildx_builder()" in script
        assert (
            'docker buildx create --name "$builder" --driver docker-container --use'
            in script
        )
        assert "docker buildx inspect --bootstrap" in script
        assert "image_input_hash()" in script
        assert "image_cache_ready()" in script
        assert 'local manifest="$repo_root/.cache/rocq-models/$target.sha"' in script
        assert 'log INFO "$target image cache hit $tag"' in script
        assert 'local cache_dir="$repo_root/.cache/rocq-models/$target"' in script
        assert '"$target.cache-from=type=local,src=$cache_dir"' in script
        assert '"$target.cache-to=type=local,dest=$cache_next,mode=max"' in script
        assert 'FIDO_UID="$(id -u)"' in script
        assert 'FIDO_GID="$(id -g)"' in script
        assert 'FIDO_USER="$(id -un)"' in script
        assert 'FIDO_HOME="$HOME"' in script
        assert 'build_image fido "$image"' in script
        assert 'build_image fido-dev "$dev_image"' in script
        assert "warm_images()" in script
        assert (
            "warm_targets=(rocq-image format lint typecheck generated-typecheck "
            "test fido fido-dev)"
        ) in script
        assert "warm_input_hash()" in script
        assert "warm_cache_ready()" in script
        assert 'local manifest="$repo_root/.cache/rocq-models/warm.sha"' in script
        assert 'log INFO "warm cache hit"' in script
        assert 'mkdir -p "$repo_root/.cache/rocq-models/context/_build"' in script
        assert (
            'git -C "$repo_root" ls-files --cached --others --exclude-standard'
            in script
        )
        assert 'for target in "${warm_targets[@]}"; do' in script
        assert "docker buildx bake" in script
        assert "warm" in script

    def test_mounts_runtime_inputs_without_mounting_whole_home(self) -> None:
        script = FIDO.read_text()

        assert "--network host" in script
        assert "--interactive" in script
        assert '--volume "$HOME:$HOME"' not in script
        assert '--volume "$repo_root:/workspace"' in script
        assert '--volume "$HOME/workspace:$HOME/workspace"' in script
        assert '--volume "$HOME/log:$HOME/log"' not in script
        assert '--volume "$secret:/run/secrets/fido-secret:ro"' in script
        assert 'add_mount_if_exists "$HOME/.claude"' in script
        assert 'add_mount_if_exists "$HOME/.claude.json"' in script
        assert 'add_mount_if_exists "$HOME/.config/gh"' in script
        assert 'add_mount_if_exists "$HOME/.cache/copilot"' in script
        assert 'chmod 600 "$secret"' in script

    def test_only_up_requires_webhook_secret(self) -> None:
        script = FIDO.read_text()

        assert "need_secret=0" in script
        assert 'elif [ "$need_secret" = "1" ]; then' in script
        assert "need_secret=1" in script

    def test_arbitrary_commands_run_through_dev_image(self) -> None:
        script = FIDO.read_text()

        assert (
            '*)\n    build_image fido-dev "$dev_image"\n    run_container "$@"'
            in script
        )
        assert (
            "Any other command is passed through to `uv run` unchanged."
            in (REPO / "src" / "fido" / "fido_help.py").read_text()
        )


class TestModelDockerfile:
    def test_bake_fido_target_hides_dockerfile_path(self) -> None:
        bake = BAKE.read_text()
        script = FIDO.read_text()

        assert 'target "fido"' in bake
        assert 'target "fido-dev"' in bake
        assert 'target "rocq-image"' in bake
        assert 'target "format"' in bake
        assert 'target "lint"' in bake
        assert 'target "typecheck"' in bake
        assert 'target "generated-typecheck"' in bake
        assert 'target "test"' in bake
        assert 'group "ci"' in bake
        assert 'group "warm"' in bake
        assert 'dockerfile = "models/Dockerfile"' in bake
        assert 'dockerfile = "Dockerfile"' in bake
        assert 'target = "fido"' in bake
        assert 'target = "fido-dev"' in bake
        assert 'target = "format"' in bake
        assert 'target = "test"' in bake
        assert 'rocq_image = "target:rocq-image"' in bake
        assert 'rocq_models_cache = ".cache/rocq-models/context"' in bake
        assert (
            'targets = ["format", "lint", "typecheck", "generated-typecheck", "test"]'
            in bake
        )
        assert (
            'targets = ["format", "lint", "typecheck", "generated-typecheck", '
            '"test", "fido", "fido-dev"]'
        ) in bake
        assert 'output = ["type=docker"]' in bake
        assert "FIDO_DEV_IMAGE" in bake
        assert "FIDO_UID = FIDO_UID" in bake
        assert "FIDO_GID = FIDO_GID" in bake
        assert "FIDO_USER = FIDO_USER" in bake
        assert "FIDO_HOME = FIDO_HOME" in bake
        assert "--file" not in script

    def test_fido_runtime_installs_prod_python_and_node_tools(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "FROM python-base AS fido-python-prod" in dockerfile
        assert "FROM fido-python-prod AS fido-python-dev" in dockerfile
        assert "FROM fido-python-prod AS fido-base" in dockerfile
        assert "COPY .python-version pyproject.toml uv.lock ./" in dockerfile
        assert "COPY . ." not in dockerfile
        assert "FROM fido-base AS node-tools" in dockerfile
        assert "FROM fido-python-dev AS fido-dev" in dockerfile
        assert "COPY package.json package-lock.json ./" in dockerfile
        assert "RUN --mount=type=cache,target=/root/.npm" in dockerfile
        assert "npm ci --omit=dev --ignore-scripts" in dockerfile
        assert "npm rebuild @anthropic-ai/claude-code" in dockerfile
        assert (
            "COPY --from=node-tools /workspace/node_modules "
            "/opt/fido-node-tools/node_modules"
        ) in dockerfile
        assert "gh_${GH_VERSION}_linux_amd64.deb" in dockerfile
        assert "ln -sf /opt/fido-node-tools/node_modules/.bin/claude" in dockerfile
        assert "ln -sf /opt/fido-node-tools/node_modules/.bin/copilot" in dockerfile
        assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
        assert "uv sync --frozen --no-install-project" in dockerfile

    def test_rocq_image_install_is_keyed_by_opam_inputs(self) -> None:
        dockerfile = (REPO / "rocq-python-extraction" / "Dockerfile").read_text()

        assert (
            "COPY --chown=opam:opam rocq-python-extraction.opam dune-project ./"
            in dockerfile
        )
        assert (
            "opam install -y rocq-core.9.2.0 rocq-stdlib.9.1.0 dune.3.21.1"
            in dockerfile
        )

    def test_rocq_extract_targets_have_explicit_host_inputs(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "FROM ${ROCQ_IMAGE} AS rocq-plugin-source" in dockerfile
        assert "RUN chown opam:opam /workspace" in dockerfile
        assert "USER opam" in dockerfile
        assert "COPY --chown=opam:opam dune-workspace ./" in dockerfile
        assert "rocq-python-extraction/g_python_extraction.mlg" in dockerfile
        assert "FROM rocq-plugin-source AS extract" in dockerfile
        assert (
            "COPY --chown=opam:opam models/dune-project models/dune models/*.v models/"
            in dockerfile
        )
        assert "FROM rocq-plugin-source AS test-extract" in dockerfile
        assert "rocq-python-extraction/test/*.v" in dockerfile
        assert "rocq-python-extraction/test/generated_pytest_targets.txt" in dockerfile
        assert "rocq-python-extraction/test/generated_pyright_targets.txt" in dockerfile

    def test_python_checks_have_explicit_host_inputs(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "FROM python-deps AS python-check-base" in dockerfile
        assert (
            "COPY .dockerignore build docker-bake.hcl fido package.json "
            "package-lock.json pyproject.toml pyrightconfig.json uv.lock ./"
        ) in dockerfile
        assert "COPY .githooks/pre-commit .githooks/pre-commit" in dockerfile
        assert "COPY models/Dockerfile models/Dockerfile" in dockerfile
        assert "COPY src src" in dockerfile
        assert "COPY tests tests" in dockerfile
        assert (
            "COPY rocq-python-extraction/test/*.py rocq-python-extraction/test/"
            in dockerfile
        )

    def test_fido_runtime_uses_host_uid_gid_build_args(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "ARG FIDO_UID=1000" in dockerfile
        assert "ARG FIDO_GID=1000" in dockerfile
        assert "ARG FIDO_USER=fido" in dockerfile
        assert "ARG FIDO_HOME=/home/fido" in dockerfile
        assert 'useradd --uid "$FIDO_UID" --gid "$FIDO_GID"' in dockerfile

    def test_ci_is_parallel_meta_target(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "FROM python-check-base AS format" in dockerfile
        assert "FROM python-check-base AS lint" in dockerfile
        assert "FROM python-check-base AS typecheck" in dockerfile
        assert "FROM python-check-base AS generated-typecheck" in dockerfile
        assert "FROM python-check-base AS test" in dockerfile
        assert "FROM scratch AS ci" not in dockerfile
        assert "FROM scratch AS warm" not in dockerfile
        assert "touch /tmp" not in dockerfile
        assert "-ready" not in dockerfile
