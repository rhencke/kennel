.PHONY: models

# Extract Rocq models to Python and deposit in kennel/models_generated/.
# Requires: docker (with buildx), uv
models:
	models/build.sh
