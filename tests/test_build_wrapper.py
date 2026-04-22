from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIDO = REPO / "fido"
BAKE = REPO / "docker-bake.hcl"
PRE_COMMIT = REPO / ".githooks" / "pre-commit"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
CI_GENERATOR = REPO / "tools" / "gen_workflows.py"
LSP_CONFIG = REPO / ".lsp.json"


def run_build(
    tmp_path: Path, output: Path, *command: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(FIDO), "make-rocq", "--smart-output", str(output), "--", *command],
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
        script = FIDO.read_text()

        assert "docker buildx bake" in script
        assert "docker cp" not in script
        assert "uv run" not in script
        assert "dune build" not in script

    def test_make_rocq_uses_bake_target_and_smart_output(self) -> None:
        script = FIDO.read_text()
        bake = BAKE.read_text()

        assert "buildx_driver()" in script
        assert "warning: docker buildx inspect failed" in script
        assert '--smart-output "$repo_root/src/fido/rocq"' in script
        assert "docker buildx bake \\" in script
        assert "make-rocq" in script
        assert "image_oci" not in script
        assert "image_cache_next" not in script
        assert "build_cache_next" not in script
        assert 'target "make-rocq"' in bake
        assert 'target = "export"' in bake
        assert 'output = ["type=local,dest=."]' in bake
        assert 'rocq_image = "target:rocq-image"' in bake
        assert 'rocq_models_cache = ".cache/rocq-models/context"' in bake

    def test_scripts_are_executable(self) -> None:
        assert os.access(FIDO, os.X_OK)

    def test_lsp_config_points_to_fido_rocq_lsp(self) -> None:
        config = json.loads(LSP_CONFIG.read_text())

        rocq = config["languageServers"]["rocq"]
        assert rocq["command"] == "./fido"
        assert rocq["args"] == ["rocq-lsp"]
        assert rocq["extensions"] == [".v"]

    def test_pre_commit_runs_ci_and_runtime_smoke(self) -> None:
        script = PRE_COMMIT.read_text()

        assert "./fido ci" in script

    def test_ci_workflow_is_generated_for_self_hosted_runner(self) -> None:
        workflow = CI_WORKFLOW.read_text()
        generator = CI_GENERATOR.read_text()

        assert "Generated by ./fido gen-workflows" in workflow
        assert "Refresh with: ./fido gen-workflows" in workflow
        assert "Buildx local cache scopes discovered from bake:" in workflow
        assert "schedule:" not in workflow
        assert "cron:" not in workflow
        assert "runs-on: self-hosted" in workflow
        assert "FIDO_BUILDX_CACHE_BACKEND" not in workflow
        assert "ubuntu-latest" not in workflow
        assert "docker/setup-buildx-action" not in workflow
        assert "Compute build cache bucket" not in workflow
        assert "518400" not in workflow
        assert "steps.build-cache.outputs.bucket" not in workflow
        assert "github.run_id" not in workflow
        assert "github.run_attempt" not in workflow
        assert "Compute build input cache key" not in workflow
        assert "tar --sort=name --mtime='UTC 1970-01-01'" not in workflow
        assert "sha256sum" not in workflow
        assert "hashFiles(" not in workflow
        assert "id: build-input" not in workflow
        assert "steps.build-input.outputs" not in workflow
        assert "steps.build-input.outputs.key" not in workflow
        assert "parse_dockerfile" in generator
        assert "target_inputs" in generator
        assert "render_build_graph" in generator
        assert "tools/build_graph.sh" in workflow
        assert "Check generated workflow files" in workflow
        assert (
            "git diff --exit-code -- .github/workflows/ci.yml tools/build_graph.sh"
            in workflow
        )
        assert "actions/cache" not in workflow
        assert "reproducible-containers/buildkit-cache-dance" not in workflow
        assert "Restore rocq-model" not in workflow
        assert "Save rocq-model" not in workflow
        assert "Cache fido-" not in workflow
        assert "Inject fido-" not in workflow
        assert "Prune BuildKit cache" not in workflow
        assert "docker buildx prune --builder fido" not in workflow
        assert '"rocq-model-image":' not in generator
        assert '".cache/rocq-models/image"' not in generator
        assert "docker buildx" in generator
        assert "bake" in generator
        for target in (
            "fido",
            "format",
            "generated-typecheck",
            "lint",
            "make-rocq",
            "rocq-image",
            "test-rocq-generated",
            "test-unit",
            "typecheck",
        ):
            assert target in workflow
        assert "Restore buildkit cache mounts" not in workflow
        assert "key: buildkit-mounts-${{ runner.os }}-" not in workflow
        assert "cache-dir: .cache/buildkit-mounts" not in workflow
        assert "dockerfile: models/Dockerfile" not in workflow


class TestFidoLauncher:
    def test_maps_friendly_commands_to_project_scripts(self) -> None:
        script = FIDO.read_text()

        assert "help)" in script
        assert "run_fido_cli_image fido-help" in script
        assert "up)" in script
        assert "supervise_up fido --secret-file /run/secrets/fido-secret" in script
        assert "make-rocq)" in script
        assert "make_rocq_models" in script
        assert "ci)" in script
        assert "ci_images" in script
        assert "gen-workflows)" in script
        assert (
            "docker buildx bake --print ci fido-test make-rocq > .cache/bake-plan.json"
            in script
        )
        assert (
            'python3 "$repo_root/tools/gen_workflows.py" --plan .cache/bake-plan.json'
            in script
        )
        assert "prune)" in script
        assert "prune_buildkit" in script
        assert "status)" in script
        assert "run_fido_cli_image_quiet --no-sync python -m fido.status" in script
        assert "task)" in script
        assert "run_fido_cli_image fido-task" in script
        assert "sync-tasks)" in script
        assert "run_fido_cli_image fido-sync-tasks" in script
        assert "rocq-lsp)" in script
        assert "run_rocq_lsp_image -m fido.rocq_lsp --stdio" in script
        assert "lsp)" in script
        assert "run_rocq_lsp_image -m fido.rocq_lsp" in script
        assert "run_rocq_lsp_image()" in script
        assert "run_fido_test_python_image()" in script

    def test_supervises_foreground_container_and_down_stops_by_name(self) -> None:
        script = FIDO.read_text()
        removed_flag = "--detach"

        assert removed_flag not in script
        assert "supervise_up()" in script
        assert "--rm" in script
        assert "fido_log=${FIDO_LOG:-$HOME/log/fido.log}" in script
        assert "redirect_up_logs()" in script
        assert 'exec >>"$fido_log" 2>&1' in script
        assert "prune_on_restart=${FIDO_PRUNE_ON_RESTART:-1}" in script
        assert "prune_keep_storage=${FIDO_BUILDKIT_KEEP_STORAGE:-24gb}" in script
        assert "prune_restart_buildkit_async()" in script
        assert "docker buildx prune \\" in script
        assert '--builder "$builder"' in script
        assert '--keep-storage "$prune_keep_storage"' in script
        assert "pruning buildkit cache action=manual" in script
        assert "pruning buildkit cache action=async pid=$!" in script
        assert "pruned buildkit cache action=done" in script
        assert "named_run=0" in script
        assert "named_run=1" in script
        assert 'if [ "$named_run" = "1" ]; then' in script
        assert 'docker rm -f "$container"' not in script
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

    def test_builds_runtime_and_test_images_with_host_identity(self) -> None:
        script = FIDO.read_text()

        assert "docker buildx bake" in script
        assert "run_with_bake_env()" in script
        assert "test_image=${FIDO_TEST_IMAGE:-fido-test:local}" in script
        assert "builder=${FIDO_BUILDX_BUILDER:-fido}" in script
        assert "ensure_buildx_builder()" in script
        assert (
            'docker buildx create --name "$builder" --driver docker-container --use'
            in script
        )
        assert "docker buildx inspect --bootstrap" in script
        assert "image_input_hash()" not in script
        assert "target_input_hash()" in script
        assert "tools/build_graph.sh" in script
        assert "image_cache_ready()" in script
        assert 'local manifest="$repo_root/.cache/rocq-models/$target.sha"' in script
        assert 'log INFO "$target image cache hit $tag"' in script
        assert '"$target.cache-from=type=local,src=$cache_dir"' not in script
        assert '"$target.cache-to=type=local,dest=$cache_next,mode=max"' not in script
        assert 'FIDO_UID="$(id -u)"' in script
        assert 'FIDO_GID="$(id -g)"' in script
        assert 'FIDO_USER="$(id -un)"' in script
        assert 'FIDO_HOME="$HOME"' in script
        assert "run_fido_image()" in script
        assert "run_fido_test_image()" in script
        assert "run_fido_cli_image()" in script
        assert "ci_images()" in script
        assert "ci_targets=(" not in script
        assert "bake_target_names()" not in script
        assert "fido_build_targets_for_group ci" in script
        assert "buildx_cache_backend()" not in script
        assert "FIDO_BUILDX_CACHE_BACKEND" not in script
        assert "GITHUB_ACTIONS" not in script
        assert "type=gha" not in script
        assert "ci_input_hash()" not in script
        assert "ci_cache_ready()" in script
        assert 'mkdir -p "$repo_root/.cache/rocq-models/ci"' in script
        assert 'log INFO "ci cache hit"' in script
        assert 'mkdir -p "$repo_root/.cache/rocq-models/context/_build"' in script
        assert "target_input_files()" in script
        assert 'target_hashes+=("$target $(target_input_hash "$target")")' in script
        assert '--set "fido.output=type=cacheonly"' in script
        assert "exporting buildx cache target=" not in script
        assert "docker buildx bake" in script
        assert "ci" in script
        assert 'git -C "$repo_root" ls-files --cached --others' not in script
        assert "target_input_files make-rocq" in script

    def test_mounts_runtime_inputs_without_mounting_whole_home(self) -> None:
        script = FIDO.read_text()

        assert "--network host" in script
        assert "--interactive" in script
        assert '--env "PYTHONPATH=/workspace/src"' in script
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

    def test_supported_dev_tools_run_through_test_image(self) -> None:
        script = FIDO.read_text()
        help_text = (REPO / "src" / "fido" / "fido_help.py").read_text()

        assert "ruff|pyright|pytest)" in script
        assert 'run_fido_test_image "$@"' in script
        assert 'echo "unsupported fido command: $command" >&2' in script
        assert "Any other command is passed through" not in help_text
        assert "rocq-lsp" in help_text
        assert "lsp" in help_text
        assert "ruff" in help_text
        assert "pyright" in help_text
        assert "pytest" in help_text


class TestModelDockerfile:
    def test_bake_fido_target_hides_dockerfile_path(self) -> None:
        bake = BAKE.read_text()
        script = FIDO.read_text()
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert 'target "fido"' in bake
        assert 'target "fido-test"' in bake
        assert 'target "make-rocq"' in bake
        assert 'target "rocq-image"' in bake
        assert 'target "format"' in bake
        assert 'target "lint"' in bake
        assert 'target "typecheck"' in bake
        assert 'target "generated-typecheck"' in bake
        assert 'target "test-unit"' in bake
        assert 'target "test-rocq-generated"' in bake
        assert 'group "ci"' in bake
        assert 'dockerfile = "models/Dockerfile"' in bake
        assert 'dockerfile = "Dockerfile"' in bake
        assert 'target = "fido"' in bake
        assert 'target = "fido-test"' in bake
        assert 'target = "export"' in bake
        assert 'target = "format"' in bake
        assert 'target = "test-unit"' in bake
        assert 'target = "test-rocq-generated"' in bake
        assert 'rocq_image = "target:rocq-image"' in bake
        assert 'rocq_models_cache = ".cache/rocq-models/context"' in bake
        assert ".lsp.json" in dockerfile
        assert (
            'targets = ["format", "lint", "typecheck", "generated-typecheck", '
            '"test-unit", "test-rocq-generated", "fido", "rocq-repl"]' in bake
        )
        assert 'output = ["type=docker"]' in bake
        assert "FIDO_TEST_IMAGE" in bake
        assert "FIDO_UID = FIDO_UID" in bake
        assert "FIDO_GID = FIDO_GID" in bake
        assert "FIDO_USER = FIDO_USER" in bake
        assert "FIDO_HOME = FIDO_HOME" in bake
        assert "docker buildx bake" in script

    def test_fido_runtime_installs_prod_python_and_node_tools(self) -> None:
        dockerfile = (REPO / "models" / "Dockerfile").read_text()

        assert "FROM python-base AS fido-python-prod" in dockerfile
        assert "FROM fido-python-prod AS fido-python-dev" in dockerfile
        assert "FROM node:24-bookworm-slim AS node-runtime" in dockerfile
        assert "FROM fido-python-prod AS fido-base" in dockerfile
        assert "COPY .python-version pyproject.toml uv.lock ./" in dockerfile
        assert "COPY . ." not in dockerfile
        assert "nodejs npm" not in dockerfile
        assert "FROM node-runtime AS node-tools" in dockerfile
        assert "FROM fido AS fido-test" in dockerfile
        assert "COPY package.json package-lock.json ./" in dockerfile
        assert "RUN --mount=type=cache,id=fido-npm,target=/root/.npm" in dockerfile
        assert "npm ci --omit=dev --ignore-scripts" in dockerfile
        assert "npm rebuild @anthropic-ai/claude-code" in dockerfile
        assert (
            "COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node"
            in dockerfile
        )
        assert (
            "COPY --from=node-tools /workspace/node_modules "
            "/opt/fido-node-tools/node_modules"
        ) in dockerfile
        assert "gh_${GH_VERSION}_linux_amd64.deb" in dockerfile
        assert "ln -sf /opt/fido-node-tools/node_modules/.bin/claude" in dockerfile
        assert "ln -sf /opt/fido-node-tools/node_modules/.bin/copilot" in dockerfile
        assert "uv sync --frozen --no-dev --no-install-project" in dockerfile
        assert "uv sync --frozen --no-install-project" in dockerfile
        assert (
            "chmod -R a+rwX /opt/fido-node-tools /opt/fido-venv /opt/uv-python"
            not in dockerfile
        )
        assert (
            'chown -R "$FIDO_UID:$FIDO_GID" /opt/fido-venv /opt/uv-python'
            not in dockerfile
        )

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
            "COPY .dockerignore .lsp.json .python-version docker-bake.hcl dune-workspace fido package.json "
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
        assert "FROM python-deps AS python-test-base" in dockerfile
        assert (
            "rocq-python-extraction/META.rocq-python-extraction.template" in dockerfile
        )
        assert "rocq-python-extraction/g_python_extraction.mlg" in dockerfile

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
        assert "FROM python-test-base AS generated-typecheck" in dockerfile
        assert (
            "COPY rocq-python-extraction/test/pyright_*.py rocq-python-extraction/test/"
        ) in dockerfile
        assert "FROM python-test-base AS test-unit" in dockerfile
        assert "FROM python-test-base AS test-rocq-generated" in dockerfile
        assert "uv run tests-unit" in dockerfile
        assert "uv run tests-rocq-generated" in dockerfile
        assert "FROM scratch AS ci" not in dockerfile
        assert "touch /tmp" not in dockerfile
        assert "-ready" not in dockerfile
