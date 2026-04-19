.PHONY: build test clean docker-build docker-test

build:
	dune build

# Round-trip assertions and syntax checks live in the dune runtest rules
# (see the dune file) so that each test is defined in exactly one place.
# This target delegates entirely to dune — do NOT add assertions here.
test:
	dune runtest

clean:
	dune clean

# ---------------------------------------------------------------------------
# Local Docker targets — mirrors the CI build so your dev environment matches
# exactly what GitHub Actions runs.
#
#   make docker-build   — build the image from the local Dockerfile (uses
#                          Docker's own layer cache; no GHA credentials needed)
#   make docker-test    — build the image then run the full test suite inside
#                          the container, mounting the repo root read-only
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
	docker run --rm -v "$(CURDIR)/..:/src:ro" rocq-python-extraction:ci \
	  sh -c "cp -r /src /tmp/work && cd /tmp/work/rocq-python-extraction && opam exec -- make test"
