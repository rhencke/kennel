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
	_build/default/list_add_one.py \
	_build/default/mylist_is_empty.py \
	_build/default/bintree_is_leaf.py \
	_build/default/roseforest_is_empty.py \
	_build/default/mforest_is_empty.py \
	_build/default/myopt_flatten.py \
	_build/default/even_depth.py \
	_build/default/is_even.py \
	_build/default/ntree_is_leaf.py

test: build
	@for f in $(PY_ARTIFACTS); do \
	  echo "checking $$f ..."; \
	  python3 -m py_compile $$f || exit 1; \
	done
	@echo "All extracted .py files are syntactically valid."
	@python3 -c "\
exec(open('_build/default/even_depth.py').read()); \
assert even_depth(EvenO()) == 0, \
  'even_depth(EvenO()): got ' + repr(even_depth(EvenO())); \
assert even_depth(EvenS(OddS(EvenO()))) == 2, \
  'even_depth depth-2: got ' + repr(even_depth(EvenS(OddS(EvenO())))); \
assert even_depth(EvenS(OddS(EvenS(OddS(EvenO()))))) == 4, \
  'even_depth depth-4: got ' + repr(even_depth(EvenS(OddS(EvenS(OddS(EvenO())))))); \
assert isinstance(EvenO(), Even), 'EvenO() must be instance of Even'; \
assert isinstance(OddS(EvenO()), Odd), 'OddS() must be instance of Odd'; \
print('Phase 4 Even/Odd round-trip: OK')"
	@python3 -c "\
exec(open('_build/default/is_even.py').read()); \
assert is_even(0) is True,  'is_even(0): got ' + repr(is_even(0)); \
assert is_even(1) is False, 'is_even(1): got ' + repr(is_even(1)); \
assert is_even(2) is True,  'is_even(2): got ' + repr(is_even(2)); \
assert is_even(3) is False, 'is_even(3): got ' + repr(is_even(3)); \
assert is_even(4) is True,  'is_even(4): got ' + repr(is_even(4)); \
print('Phase 4 is_even round-trip: OK')"
	@python3 -c "\
exec(open('_build/default/ntree_is_leaf.py').read()); \
assert ntree_is_leaf(NLeaf()) is True,  'ntree_is_leaf(NLeaf()): got ' + repr(ntree_is_leaf(NLeaf())); \
assert ntree_is_leaf(NNode([])) is False, 'ntree_is_leaf(NNode([])): got ' + repr(ntree_is_leaf(NNode([]))); \
assert ntree_is_leaf(NNode([NLeaf()])) is False, 'ntree_is_leaf(NNode([NLeaf()])): got ' + repr(ntree_is_leaf(NNode([NLeaf()]))); \
assert isinstance(NLeaf(), NTree), 'NLeaf() must be instance of NTree'; \
assert isinstance(NNode([]), NTree), 'NNode([]) must be instance of NTree'; \
print('Phase 4 NTree nested inductive round-trip: OK')"

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
