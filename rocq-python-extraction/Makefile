.PHONY: build test clean docker-build docker-test

build:
	dune build

# After dune builds the test theory, Python Extraction side-effects land in
# _build/default/.  Validate each one is syntactically valid Python.
PY_ARTIFACTS := \
	_build/default/nat_add.py \
	_build/default/mk_pair_r.py \
	_build/default/zeros.py \
	_build/default/uint_val.py \
	_build/default/float_val.py \
	_build/default/str_val.py \
	_build/default/todo_val.py \
	_build/default/bool_not.py \
	_build/default/nat_double.py \
	_build/default/option_inc.py \
	_build/default/pair_swap.py \
	_build/default/list_add_one.py

test: build
	@for f in $(PY_ARTIFACTS); do \
	  echo "checking $$f ..."; \
	  python3 -m py_compile $$f || exit 1; \
	done
	@echo "All extracted .py files are syntactically valid."

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
