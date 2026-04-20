.PHONY: build test clean docker-build docker-test

build:
	dune build

# Build the umbrella acceptance theory. This exercises plugin loading,
# Rocq compilation, and extraction side-effects in the CI container.
# Python assertions live under pytest via `uv run tests`.
test:
	dune build test/python.vo

clean:
	dune clean

# ---------------------------------------------------------------------------
# Local Docker targets — mirrors the CI build so your dev environment matches
# exactly what GitHub Actions runs.
#
#   make docker-build   — build the image from the local Dockerfile (uses
#                          Docker's own layer cache; no GHA credentials needed)
#   make docker-test    — build the image then run the Rocq acceptance build
#                          inside the container, mounting the repo root read-only
# ---------------------------------------------------------------------------

# Build the image from the Dockerfile in this directory.  The build context is
# '.' (this directory), the same files CI sends when it does:
#   docker buildx build ... rocq-python-extraction/
docker-build:
	docker buildx build --load -t rocq-python-extraction:ci .

# Run tests inside the container.  The repo root (one level up) is mounted
# read-only at /src; the container copies it to a writable /tmp/work so dune
# can write build artifacts without touching the source tree.
docker-test: docker-build
	./run_in_docker.sh . opam exec -- make test
