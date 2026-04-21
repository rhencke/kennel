.PHONY: models

# Extract Rocq models to Python and deposit in src/fido/rocq/.
# Requires: docker with buildx.
models:
	./fido make-rocq
