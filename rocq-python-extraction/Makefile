.PHONY: build test clean

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
	_build/default/nat_double.py

test: build
	@for f in $(PY_ARTIFACTS); do \
	  echo "checking $$f ..."; \
	  python3 -m py_compile $$f || exit 1; \
	done
	@echo "All extracted .py files are syntactically valid."

clean:
	dune clean
